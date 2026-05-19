/*
 * Copyright (c) 2025 Element Creations Ltd.
 *
 * SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Element-Commercial.
 */

package io.element.android.features.messages.impl.sirenbert

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import timber.log.Timber
import java.io.File
import java.nio.LongBuffer
import java.util.concurrent.ConcurrentHashMap
import kotlin.math.exp
import kotlin.math.max

/**
 * On-device equivalent of [SirenbertApiClient].
 *
 * Same input/output as the API client (PredictResponse-shaped result), but
 * runs the bundled INT8 ONNX artifacts locally via ONNX Runtime Mobile.
 *
 * Bundled assets, in `features/messages/impl/src/main/assets/sirenbert/`:
 *   - stage1.onnx (+ optional stage1.onnx.data) -- DistilBERT k=5 INT8
 *   - stage2.onnx                                -- GRU INT8
 *   - vocab.txt                                  -- DistilBERT WordPiece vocab
 *   - tokenizer_config.json                      -- max_length, do_lower_case
 *   - manifest.json                              -- candidate label, config
 *
 * State management:
 *   - Per-room context buffer: last 5 messages in the room (both roles)
 *     used as Stage 1 input prefix.
 *   - Per-room Stage 2 buffer: rolling window of the last 50 suspect-message
 *     trigger probability vectors.
 *
 * The same SirenbertSession.bump() that resets the cache also wipes both
 * buffers via [reset].
 */
class OnDeviceSirenbertEngine private constructor(
    private val tokenizer: BertWordPieceTokenizer,
    private val stage1: OrtSession,
    private val stage2: OrtSession,
    private val env: OrtEnvironment,
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
     * Run on-device classification for a single message. Returns a struct
     * matching the FastAPI /predict response shape (the fields we actually
     * use in [SirenbertCache]).
     *
     * [conversationKey] should already include the session counter
     * (e.g. "!room:server#s2") so that a reset wipes the buffers via [reset].
     */
    fun predict(
        conversationKey: String,
        role: String,
        body: String,
    ): OnDevicePrediction {
        val buf = rooms.getOrPut(conversationKey) { RoomBuffers() }

        // 1. Stage 1 input: build [CLS] m_{t-k} [SEP] ... [SEP] m_t [SEP].
        //    For target messages we still want trigger probs because the
        //    server treats them as CONTEXT_ONLY at Stage 2 level only.
        val isSuspect = role == "S"

        if (!isSuspect) {
            // Target message: record context, return CONTEXT_ONLY.
            buf.context.addLast(role to body)
            while (buf.context.size > contextK + 1) buf.context.removeFirst()
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

        // Suspect message: build segment list (last k context msgs + this one).
        // Each segment is prefixed with its role ("S: " or "T: ") to match the
        // exact format Stage 1 was trained on (sirenbert_stage1_train.py
        // build_examples: parts.append(f"{role}: {content}"), joined by [SEP]).
        // Target messages also classify as suspect at the encoder level — the
        // server uses "S: " for the message being classified regardless.
        val segments = buildList {
            for ((ctxRole, ctxBody) in buf.context) {
                add("$ctxRole: $ctxBody")
            }
            add("S: $body")
        }

        val tokenized = tokenizer.encodeSegments(segments, maxLength)

        // 2. Stage 1 inference.
        val triggerProbs = runStage1(tokenized)

        // 3. Stage 2 inference: append this vector to the rolling buffer,
        //    pad to stage2MaxSeq, run.
        buf.stage2.addLast(triggerProbs)
        while (buf.stage2.size > stage2MaxSeq) buf.stage2.removeFirst()
        val (terminalLogit, suspiciousLogit) = runStage2(buf.stage2.toList())

        // Update the context buffer AFTER classification (Stage 1 sees the
        // prior k messages, not the one being classified).
        buf.context.addLast(role to body)
        while (buf.context.size > contextK + 1) buf.context.removeFirst()

        // 4. Convert logits to probabilities (sigmoid for binary heads).
        val scamProb = sigmoid(terminalLogit)
        val susProb = sigmoid(suspiciousLogit)

        val argmax = argmax(triggerProbs)
        val triggerName = TRIGGER_NAMES[argmax]

        // For a binary terminal head, the "label" is SCAM if prob > 0.5 else
        // SUSPICIOUS if susProb > 0.5 else NON_SCAM. Mirrors the server.
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

    private fun runStage1(t: TokenizedInput): FloatArray {
        val inputIds = OnnxTensor.createTensor(
            env,
            LongBuffer.wrap(t.inputIds),
            longArrayOf(1, maxLength.toLong()),
        )
        val attentionMask = OnnxTensor.createTensor(
            env,
            LongBuffer.wrap(t.attentionMask),
            longArrayOf(1, maxLength.toLong()),
        )
        return inputIds.use {
            attentionMask.use {
                stage1.run(mapOf("input_ids" to inputIds, "attention_mask" to attentionMask)).use { results ->
                    @Suppress("UNCHECKED_CAST")
                    val logits = (results[0].value as Array<FloatArray>)[0]
                    softmax(logits)
                }
            }
        }
    }

    private fun runStage2(history: List<FloatArray>): Pair<Float, Float> {
        // Stage 2 expects [batch=1, stage2MaxSeq, 14]. Zero-pad on the LEFT
        // so the most recent timesteps are at the right (matches training).
        val flat = FloatArray(stage2MaxSeq * NUM_TRIGGERS)
        val offset = stage2MaxSeq - history.size
        for ((i, vec) in history.withIndex()) {
            val pos = (offset + i) * NUM_TRIGGERS
            System.arraycopy(vec, 0, flat, pos, NUM_TRIGGERS)
        }
        val tensor = OnnxTensor.createTensor(
            env,
            java.nio.FloatBuffer.wrap(flat),
            longArrayOf(1, stage2MaxSeq.toLong(), NUM_TRIGGERS.toLong()),
        )
        return tensor.use {
            stage2.run(mapOf("trigger_vectors" to tensor)).use { results ->
                @Suppress("UNCHECKED_CAST")
                val terminal = (results[0].value as Array<FloatArray>)[0][0]
                @Suppress("UNCHECKED_CAST")
                val suspicious = (results[1].value as Array<FloatArray>)[0][0]
                terminal to suspicious
            }
        }
    }

    companion object {
        const val NUM_TRIGGERS = 14

        // Order MUST match the offline trainer. Source of truth:
        // redefined_approach/scripts/analysis/clean_v2_length_analysis.py
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
        private const val STAGE1_ASSET = "stage1.onnx"
        private const val STAGE2_ASSET = "stage2.onnx"
        private const val VOCAB_ASSET = "vocab.txt"

        @Volatile
        private var instance: OnDeviceSirenbertEngine? = null

        /**
         * Lazily build the engine on first use. Returns null if any required
         * asset is missing (e.g. the user hasn't scp'd the bundle yet); callers
         * should fall back to the API path in that case.
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
            val env = OrtEnvironment.getEnvironment()

            // ONNX Runtime can load external .data files only when the .onnx
            // is given as a file path, not as a byte buffer. We materialise
            // the bundle into a dedicated subdir of cacheDir, preserving the
            // exact filenames so the external_data "location=stage1.onnx.data"
            // reference inside stage1.onnx still resolves.
            val cacheRoot = File(context.cacheDir, ASSET_DIR).apply { mkdirs() }
            val stage1File = copyAssetToCache(context, cacheRoot, STAGE1_ASSET)
            runCatching {
                copyAssetToCache(context, cacheRoot, "${STAGE1_ASSET}.data")
            }
            val stage2File = copyAssetToCache(context, cacheRoot, STAGE2_ASSET)
            runCatching {
                copyAssetToCache(context, cacheRoot, "${STAGE2_ASSET}.data")
            }

            val opts = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(2)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)
            }
            val stage1 = env.createSession(stage1File.absolutePath, opts)
            val stage2 = env.createSession(stage2File.absolutePath, opts)

            val vocab = context.assets.open("$ASSET_DIR/$VOCAB_ASSET").bufferedReader().use { reader ->
                BertWordPieceTokenizer.loadVocab(reader.lineSequence())
            }
            val tokenizer = BertWordPieceTokenizer(vocab = vocab)

            Timber.tag("SIRENBERT").i(
                "on-device engine initialised: stage1=%s stage2=%s vocab=%d",
                stage1File.name,
                stage2File.name,
                vocab.size,
            )
            return OnDeviceSirenbertEngine(
                tokenizer = tokenizer,
                stage1 = stage1,
                stage2 = stage2,
                env = env,
                maxLength = 512,
                contextK = 5,
                stage2MaxSeq = 50,
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
