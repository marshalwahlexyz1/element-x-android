/*
 * Copyright (c) 2025 Element Creations Ltd.
 *
 * SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Element-Commercial.
 */

package io.element.android.features.messages.impl.sirenbert

import android.content.Context
import org.tensorflow.lite.Interpreter
import timber.log.Timber
import java.io.File
import java.util.concurrent.ConcurrentHashMap
import kotlin.math.exp
import kotlin.math.max

/**
 * On-device equivalent of [SirenbertApiClient].
 *
 * Same input/output as the API client (PredictResponse-shaped result), but runs the bundled FP16
 * TFLite (LiteRT) artifacts locally. Migrated from ONNX Runtime -> LiteRT so the CV-selected model
 * (DistilBERT k=1) ships as FP16 TFLite (argmax-lossless vs PyTorch; ONNX↔TFLite verified 100%
 * trigger agreement offline).
 *
 * Bundled assets, in `features/messages/impl/src/main/assets/sirenbert/`:
 *   - stage1.tflite   -- Stage-1 encoder, FP16, FIXED input length [1, MAX_LENGTH]
 *   - stage2.tflite   -- Stage-2 GRU, per-timestep unrolled to STAGE2_UNROLL steps
 *   - vocab.txt       -- WordPiece vocab
 *
 * TFLite I/O contract (verified offline against PyTorch — do NOT reorder without re-checking):
 *   Stage 1: IN[0]=attention_mask int64 [1,L], IN[1]=input_ids int64 [1,L]; OUT[0]=logits f32 [1,14].
 *   Stage 2: IN[0]=trigger_vectors f32 [1,14,50] (trigger-major, timestep last), TRAILING-padded
 *            (real vectors at timesteps 0..T-1); OUT[0]=suspicious [1,50], OUT[1]=terminal [1,50];
 *            read index T-1. Trailing-pad + read-T-1 == the dynamic-prefix reference GRU exactly.
 *
 * State management: per-room context buffer (last k messages) + per-room Stage-2 buffer (last 50
 * suspect trigger vectors). SirenbertSession.bump() wipes both via [reset].
 */
class OnDeviceSirenbertEngine private constructor(
    private val tokenizer: BertWordPieceTokenizer,
    private val stage1: Interpreter,
    private val stage2: Interpreter,
    private val maxLength: Int,
    private val contextK: Int,
    private val stage2MaxSeq: Int,
) {
    private data class RoomBuffers(
        val context: ArrayDeque<Pair<String, String>> = ArrayDeque(), // (role, body)
        val stage2: ArrayDeque<FloatArray> = ArrayDeque(),             // last 50 trigger vecs
    )

    private val rooms = ConcurrentHashMap<String, RoomBuffers>()

    /**
     * Run on-device classification for a single message. Returns a struct matching the FastAPI
     * /predict response shape (the fields [SirenbertCache] uses).
     *
     * [conversationKey] should already include the session counter (e.g. "!room:server#s2") so a
     * reset wipes the buffers via [reset].
     */
    fun predict(
        conversationKey: String,
        role: String,
        body: String,
    ): OnDevicePrediction {
        val buf = rooms.getOrPut(conversationKey) { RoomBuffers() }

        val isSuspect = role == "S"
        if (!isSuspect) {
            // Target message: record context, return CONTEXT_ONLY.
            buf.context.addLast(role to body)
            while (buf.context.size > contextK) buf.context.removeFirst()
            return OnDevicePrediction(
                label = "CONTEXT_ONLY",
                conversationLabel = "CONTEXT_ONLY",
                messageTrigger = "CONTEXT_ONLY",
                suspiciousProbability = 0f,
                scamProbability = 0f,
                convSuspiciousProbability = 0f,
                convScamProbability = 0f,
                storedAsContextOnly = true,
            )
        }

        // Suspect message: build role-prefixed segments (last k context msgs + this one), exactly the
        // format Stage 1 was trained on (build_examples: f"{role}: {content}", joined by [SEP]).
        val segments = buildList {
            for ((ctxRole, ctxBody) in buf.context) add("$ctxRole: $ctxBody")
            add("S: $body")
        }
        val tokenized = tokenizer.encodeSegments(segments, maxLength)

        // Stage 1 -> trigger probs.
        val triggerProbs = runStage1(tokenized)

        // Stage 2: append to rolling buffer (cap 50), run per-timestep unroll, read the T-1 output.
        buf.stage2.addLast(triggerProbs)
        while (buf.stage2.size > stage2MaxSeq) buf.stage2.removeFirst()
        val (terminalLogit, suspiciousLogit) = runStage2(buf.stage2.toList())

        // Context buffer updates AFTER classification (Stage 1 sees the prior k, not the current one).
        buf.context.addLast(role to body)
        while (buf.context.size > contextK) buf.context.removeFirst()

        val scamProb = sigmoid(terminalLogit)
        val susProb = sigmoid(suspiciousLogit)
        val triggerName = TRIGGER_NAMES[argmax(triggerProbs)]
        val label = when {
            scamProb >= 0.5f -> "SCAM"
            susProb >= 0.5f -> "SUSPICIOUS"
            else -> "NON_SCAM"
        }

        return OnDevicePrediction(
            label = label,
            conversationLabel = label,
            messageTrigger = triggerName,
            suspiciousProbability = susProb,
            scamProbability = scamProb,
            convSuspiciousProbability = susProb,
            convScamProbability = scamProb,
            storedAsContextOnly = false,
        )
    }

    /** Drop the per-room buffers (e.g. on SIRENBERT_RESET adb broadcast). */
    fun reset() {
        rooms.clear()
    }

    /**
     * SIRENBERT cross-app handoff: ensure a per-conversation buffer exists for [conversationKey] so
     * subsequent messages continue the tracking sequence. Called when a sirenbert://track deeplink is
     * opened after a handoff from another SIRENBERT client. Idempotent; never carries content across apps.
     */
    fun seedConversation(conversationKey: String) {
        val created = !rooms.containsKey(conversationKey)
        rooms.getOrPut(conversationKey) { RoomBuffers() }
        Timber.tag("SIRENBERT").i(
            "handoff seed conv=%s created=%s trackedConvs=%d",
            conversationKey, created, rooms.size,
        )
    }

    private fun runStage1(t: TokenizedInput): FloatArray {
        // Verified TFLite input ORDER: IN[0]=attention_mask, IN[1]=input_ids (onnx2tf sorts inputs
        // alphabetically). Both int64 [1, maxLength]; the tokenizer already pads/truncates to maxLength.
        val attentionMask = arrayOf(t.attentionMask)          // [1, maxLength] int64
        val inputIds = arrayOf(t.inputIds)                    // [1, maxLength] int64
        val logits = Array(1) { FloatArray(NUM_TRIGGERS) }    // OUT[0] = [1, 14] f32
        stage1.runForMultipleInputsOutputs(arrayOf(attentionMask, inputIds), mapOf(0 to logits))
        return softmax(logits[0])
    }

    private fun runStage2(history: List<FloatArray>): Pair<Float, Float> {
        val seqLen = history.size.coerceAtMost(stage2MaxSeq).coerceAtLeast(1)
        // Input [1, 14, 50] f32, trigger-major with timestep LAST; TRAILING-pad (real at 0..T-1).
        // trailing-pad + read index T-1 reproduces the dynamic-prefix reference GRU EXACTLY.
        val input = Array(1) { Array(NUM_TRIGGERS) { FloatArray(STAGE2_UNROLL) } }
        for (ti in 0 until seqLen) {
            val vec = history[ti]
            for (f in 0 until NUM_TRIGGERS) input[0][f][ti] = vec[f]
        }
        // Verified output ORDER (swapped vs export names): OUT[0]=suspicious, OUT[1]=terminal, each [1,50].
        val suspiciousOut = Array(1) { FloatArray(STAGE2_UNROLL) }
        val terminalOut = Array(1) { FloatArray(STAGE2_UNROLL) }
        stage2.runForMultipleInputsOutputs(arrayOf(input), mapOf(0 to suspiciousOut, 1 to terminalOut))
        val idx = seqLen - 1
        return terminalOut[0][idx] to suspiciousOut[0][idx]
    }

    companion object {
        const val NUM_TRIGGERS = 14
        private const val STAGE2_UNROLL = 50                  // Stage-2 export unrolls to a fixed 50 steps

        // Order MUST match the offline trainer (IDX_TO_TRIGGER in sirenbert_stage1_train.py).
        val TRIGGER_NAMES = arrayOf(
            "SmallTalk_Maintenance",
            "WrongNumber_Intro",
            "Introduction_Opener",
            "Rapport_SelfDisclosure",
            "Affectionate_Language",
            "Credential_Positioning",
            "Investment_Pitch",
            "Tech_Simulation",
            "Platform_Migration",
            "Love_Bombing",
            "Financial_Demand",
            "Urgency_Pressure",
            "Crisis_Fabrication",
            "Emotional_Manipulation",
        )

        private const val ASSET_DIR = "sirenbert"
        private const val STAGE1_ASSET = "stage1.tflite"
        private const val STAGE2_ASSET = "stage2.tflite"
        private const val VOCAB_ASSET = "vocab.txt"

        // CV-selected on-device pick: DistilBERT k=1 -> fixed Stage-1 length 256, context k=1.
        // (ModernBERT-8k k=10 would use MAX_LENGTH=1024, CONTEXT_K=10 with its own asset bundle.)
        private const val MAX_LENGTH = 256
        private const val CONTEXT_K = 1
        private const val STAGE2_MAX_SEQ = 50

        @Volatile
        private var instance: OnDeviceSirenbertEngine? = null

        /**
         * Lazily build the engine on first use. Returns null if any required asset is missing (e.g. the
         * user hasn't dropped the bundle into assets/sirenbert/ yet); callers fall back to the API path.
         */
        fun getOrNull(context: Context): OnDeviceSirenbertEngine? {
            instance?.let { return it }
            return synchronized(this) {
                instance ?: try {
                    val built = build(context)
                    instance = built
                    built
                } catch (t: Throwable) {
                    Timber.tag("SIRENBERT").w(t, "on-device engine init failed; fall back to API")
                    null
                }
            }
        }

        private fun build(context: Context): OnDeviceSirenbertEngine {
            // LiteRT reads the model from a file; materialise the single-file .tflite bundle into cacheDir.
            val cacheRoot = File(context.cacheDir, ASSET_DIR).apply { mkdirs() }
            val stage1File = copyAssetToCache(context, cacheRoot, STAGE1_ASSET)
            val stage2File = copyAssetToCache(context, cacheRoot, STAGE2_ASSET)

            val opts = Interpreter.Options().apply { numThreads = 2 }
            val stage1 = Interpreter(stage1File, opts)
            val stage2 = Interpreter(stage2File, opts)

            val vocab = context.assets.open("$ASSET_DIR/$VOCAB_ASSET").bufferedReader().use { reader ->
                BertWordPieceTokenizer.loadVocab(reader.lineSequence())
            }
            val tokenizer = BertWordPieceTokenizer(vocab = vocab)

            Timber.tag("SIRENBERT").i(
                "on-device engine initialised (LiteRT): stage1=%s stage2=%s vocab=%d maxLen=%d k=%d",
                stage1File.name, stage2File.name, vocab.size, MAX_LENGTH, CONTEXT_K,
            )
            return OnDeviceSirenbertEngine(
                tokenizer = tokenizer,
                stage1 = stage1,
                stage2 = stage2,
                maxLength = MAX_LENGTH,
                contextK = CONTEXT_K,
                stage2MaxSeq = STAGE2_MAX_SEQ,
            )
        }

        private fun copyAssetToCache(context: Context, cacheRoot: File, assetName: String): File {
            val out = File(cacheRoot, assetName)
            if (!out.exists() || out.length() == 0L) {
                context.assets.open("$ASSET_DIR/$assetName").use { input ->
                    out.outputStream().use { output -> input.copyTo(output) }
                }
            }
            return out
        }

        private fun softmax(logits: FloatArray): FloatArray {
            var maxLogit = Float.NEGATIVE_INFINITY
            for (v in logits) if (v > maxLogit) maxLogit = v
            var sum = 0.0
            val exps = FloatArray(logits.size)
            for (i in logits.indices) {
                val e = exp((logits[i] - maxLogit).toDouble())
                exps[i] = e.toFloat()
                sum += e
            }
            val inv = 1.0 / max(sum, 1e-12)
            for (i in exps.indices) exps[i] = (exps[i] * inv).toFloat()
            return exps
        }

        private fun sigmoid(x: Float): Float = (1.0 / (1.0 + exp(-x.toDouble()))).toFloat()

        private fun argmax(arr: FloatArray): Int {
            var best = 0
            var bestVal = arr[0]
            for (i in 1 until arr.size) {
                if (arr[i] > bestVal) {
                    bestVal = arr[i]
                    best = i
                }
            }
            return best
        }
    }
}

/** Result shape compatible with the FastAPI server's PredictResponse. */
data class OnDevicePrediction(
    val label: String?,
    val conversationLabel: String?,
    val messageTrigger: String?,
    val suspiciousProbability: Float?,
    val scamProbability: Float?,
    val convSuspiciousProbability: Float?,
    val convScamProbability: Float?,
    val storedAsContextOnly: Boolean?,
)
