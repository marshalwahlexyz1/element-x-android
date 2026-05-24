/*
 * Copyright (c) 2025 Element Creations Ltd.
 *
 * SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Element-Commercial.
 */

package io.element.android.features.messages.impl.sirenbert

import android.content.Context
import android.os.SystemClock
import org.json.JSONArray
import org.json.JSONObject
import timber.log.Timber
import java.io.File

/**
 * Bulk on-device evaluation harness.
 *
 * Drives [OnDeviceSirenbertEngine] over an eval_set.json (the same schema
 * produced by `redefined_approach/scripts/deployment/eval/build_eval_set.py`)
 * and writes a predictions_ondevice.json that the SCC-side diff script
 * can compare directly to the PyTorch checkpoint's predictions.
 *
 * Workflow (from laptop):
 *
 *   adb push eval_set.json \
 *     /sdcard/Android/data/io.element.android.x.debug/files/eval_set.json
 *   adb shell am broadcast \
 *     -a io.element.android.x.debug.SIRENBERT_BULK_REPLAY
 *
 *   # wait for completion line in logcat:
 *   #   SIRENBERT: bulk replay done -> predictions_ondevice_<ts>.json
 *
 *   adb pull \
 *     /sdcard/Android/data/io.element.android.x.debug/files/ phone_eval/
 *
 * The receiver in [SirenbertResetReceiver] handles ACTION_BULK_REPLAY and
 * delegates to [SirenbertBulkReplay.run] in a long-lived coroutine scope.
 * The receiver returns immediately; the run can take many minutes.
 *
 * Isolation: every conversation key is prefixed with "bulkeval_" + run
 * timestamp so the bulk run does not pollute the Stage 2 buffers used by
 * real Element X chats.
 */
object SirenbertBulkReplay {
    private const val INPUT_FILENAME = "eval_set.json"
    private const val TRIGGER_NUM_CLASSES = 14

    // Single-run guard: a second SIRENBERT_BULK_REPLAY broadcast while a run is
    // already in flight would spawn a concurrent run that competes for CPU
    // (latency climbs, both runs get throttled). Ignore re-triggers until done.
    private val running = java.util.concurrent.atomic.AtomicBoolean(false)

    /**
     * Run the full bulk replay once, blocking the calling coroutine until done.
     * Invoked from [SirenbertBulkReplayService] (a foreground service) so the
     * long full-test-set run is not suspended by the OS. Re-entrant calls while
     * a run is in flight are ignored by the guard.
     */
    fun runOnce(appContext: Context) {
        if (!running.compareAndSet(false, true)) {
            Timber.tag("SIRENBERT").w("bulk replay already running; ignoring re-trigger")
            return
        }
        try {
            runInternal(appContext)
        } finally {
            running.set(false)
        }
    }

    private fun runInternal(appContext: Context) {
        val dir = appContext.getExternalFilesDir(null)
        if (dir == null) {
            Timber.tag("SIRENBERT").w("bulk replay: getExternalFilesDir(null) returned null")
            return
        }
        val inFile = File(dir, INPUT_FILENAME)
        if (!inFile.exists()) {
            Timber.tag("SIRENBERT").w("bulk replay: %s missing", inFile.absolutePath)
            return
        }
        val engine = OnDeviceSirenbertEngine.getOrNull(appContext)
        if (engine == null) {
            Timber.tag("SIRENBERT").w("bulk replay: on-device engine not initialised; aborting")
            return
        }

        val tStart = SystemClock.elapsedRealtime()
        val evalRoot = JSONObject(inFile.readText(Charsets.UTF_8))
        val candidate = evalRoot.optString("candidate", "unknown")
        val datasetId = evalRoot.optString("dataset_id", "unknown")
        val convs = evalRoot.getJSONArray("conversations")
        Timber.tag("SIRENBERT").i(
            "bulk replay start: candidate=%s convs=%d input=%s",
            candidate, convs.length(), inFile.absolutePath,
        )

        // Unique prefix so bulk-replay conversation keys don't collide with
        // real Element X room ids.
        val runTs = System.currentTimeMillis()
        val keyPrefix = "bulkeval#$runTs#"

        val outConvs = JSONArray()
        var totalSuspect = 0
        var totalTarget = 0
        var totalLatencyNs = 0L

        for (i in 0 until convs.length()) {
            val conv = convs.getJSONObject(i)
            val convId = conv.getString("conv_id")
            val gtLabel = conv.optString("ground_truth_label")
            val source = conv.optString("source", null)
            val messages = conv.getJSONArray("messages")
            val conversationKey = keyPrefix + convId

            val outMsgs = JSONArray()
            var finalVerdict = "NON_SCAM"

            for (m in 0 until messages.length()) {
                val msg = messages.getJSONObject(m)
                val role = msg.getString("role")
                val body = msg.getString("body")
                val idx = msg.getInt("idx")

                val tBeforeNs = System.nanoTime()
                val pred = engine.predict(conversationKey, role, body)
                val latencyNs = System.nanoTime() - tBeforeNs

                if (role == "T") {
                    totalTarget++
                    outMsgs.put(
                        JSONObject().apply {
                            put("idx", idx)
                            put("role", "T")
                            put("verdict", "CONTEXT_ONLY")
                        }
                    )
                    continue
                }
                totalSuspect++
                totalLatencyNs += latencyNs

                val verdict = pred.label ?: "NON_SCAM"
                finalVerdict = verdict
                outMsgs.put(
                    JSONObject().apply {
                        put("idx", idx)
                        put("role", "S")
                        put("trigger_name", pred.messageTrigger ?: "")
                        put("scam_prob", (pred.scamProbability ?: 0f).toDouble())
                        put("susp_prob", (pred.suspiciousProbability ?: 0f).toDouble())
                        put("verdict", verdict)
                        put("latency_ms", latencyNs / 1_000_000.0)
                    }
                )
            }

            outConvs.put(
                JSONObject().apply {
                    put("conv_id", convId)
                    if (gtLabel.isNotEmpty()) put("ground_truth_label", gtLabel)
                    if (!source.isNullOrEmpty()) put("source", source)
                    put("final_verdict", finalVerdict)
                    put("messages", outMsgs)
                }
            )

            if ((i + 1) % 25 == 0 || i == convs.length() - 1) {
                Timber.tag("SIRENBERT").i(
                    "bulk replay: %d/%d convs, suspect=%d target=%d, mean_latency=%.1fms",
                    i + 1, convs.length(), totalSuspect, totalTarget,
                    if (totalSuspect > 0) totalLatencyNs / 1_000_000.0 / totalSuspect else 0.0,
                )
            }
        }

        // Clean up the per-room buffers we created so the engine state matches
        // pre-run for any future real-DM activity.
        engine.reset()

        val elapsedMs = SystemClock.elapsedRealtime() - tStart
        val meanLatencyMs =
            if (totalSuspect > 0) totalLatencyNs / 1_000_000.0 / totalSuspect else 0.0

        val outFile = File(dir, "predictions_ondevice_${runTs}.json")
        val root = JSONObject().apply {
            put("schema_version", "sirenbert.predictions.v1")
            put("engine", "ondevice_ort_int8")
            put("candidate", candidate)
            put("dataset_id", datasetId)
            put("run_timestamp_ms", runTs)
            put("elapsed_ms", elapsedMs)
            put("n_conversations", outConvs.length())
            put("n_suspect_messages", totalSuspect)
            put("n_target_messages", totalTarget)
            put("mean_suspect_latency_ms", meanLatencyMs)
            put("conversations", outConvs)
        }
        outFile.writeText(root.toString(2))
        Timber.tag("SIRENBERT").i(
            "bulk replay done -> %s (suspect=%d target=%d mean_latency=%.1fms elapsed=%dms)",
            outFile.absolutePath, totalSuspect, totalTarget, meanLatencyMs, elapsedMs,
        )
    }
}
