/*
 * Copyright (c) 2025 Element Creations Ltd.
 *
 * SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Element-Commercial.
 */

package io.element.android.features.messages.impl.sirenbert

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.service.notification.NotificationListenerService
import android.service.notification.StatusBarNotification
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import androidx.core.app.Person
import timber.log.Timber

/**
 * Cross-app SIRENBERT (Pattern B / "system Smart Reply" style, NO root).
 *
 * Reads incoming message notifications from ANY app on the device and feeds them to the same
 * [OnDeviceSirenbertEngine] Element X uses in-app. The user grants "Notification access" once in
 * Settings; no root, no per-app integration — exactly how system Smart Reply reads notifications.
 *
 * Role mapping (matches the engine's S/T contract — see [OnDeviceSirenbertEngine.predict]):
 *   - message from the conversation partner (the sender)      -> role "S"  (suspect; CLASSIFIED)
 *   - message from the device user (the MessagingStyle "me")  -> role "T"  (context only)
 *
 * Rich `MessagingStyle` notifications carry BOTH sides (the user's own replies are attached as the
 * "me" person), so we reconstruct the two-sided context the model was trained on. If an app posts
 * only the incoming line, the engine still runs with suspect-only context (degraded but functional).
 *
 * Caveat: for apps whose notifications omit the user's replies, use the AccessibilityService variant
 * to recover the T side from the on-screen thread.
 */
class SirenbertNotificationListenerService : NotificationListenerService() {

    // Highest message timestamp already processed per conversation, so re-posted history is skipped.
    private val lastSeenTs = HashMap<String, Long>()

    override fun onListenerConnected() {
        ensureChannel()
        Timber.tag(TAG).i("cross-app SIRENBERT notification listener connected")
    }

    override fun onNotificationPosted(sbn: StatusBarNotification) {
        val notification: Notification = sbn.notification ?: return
        // Only messaging notifications carry a MessagingStyle; everything else is ignored.
        val style = NotificationCompat.MessagingStyle
            .extractMessagingStyleFromNotification(notification)
        if (style == null) {
            Timber.tag(TAG).v("notif from %s: not MessagingStyle, skipped", sbn.packageName)
            return
        }
        val engine = OnDeviceSirenbertEngine.getOrNull(applicationContext)
        if (engine == null) {
            Timber.tag(TAG).w("notif from %s: MessagingStyle but engine is NULL (model not loaded)", sbn.packageName)
            return
        }
        Timber.tag(TAG).i("processing messaging notif from %s (%d messages)", sbn.packageName, style.messages.size)

        val convTitle = style.conversationTitle?.toString() ?: sbn.tag ?: "chat"
        // Per-conversation buffer key (package + conversation), so the engine keeps a separate
        // rolling context window per chat, and cross-app conversations stay distinct.
        val conversationKey = "${sbn.packageName}#$convTitle"
        val me: Person? = style.user

        val since = lastSeenTs[conversationKey] ?: 0L
        var newest = since
        // MessagingStyle messages are ordered oldest -> newest; only handle ones newer than seen.
        for (msg in style.messages) {
            val ts = msg.timestamp
            if (ts <= since) continue
            newest = maxOf(newest, ts)
            val body = msg.text?.toString()?.trim().orEmpty()
            if (body.isEmpty()) continue

            val fromMe = msg.person == null ||
                (me != null && (msg.person?.key == me.key || msg.person?.name == me.name))
            val role = if (fromMe) "T" else "S"

            val pred = engine.predict(conversationKey, role, body)
            if (role == "S") {
                val verdict = pred.label ?: "NON_SCAM"
                if (verdict == "SCAM" || verdict == "SUSPICIOUS") {
                    alert(sbn.packageName, conversationKey, convTitle, verdict, pred)
                }
            }
        }
        lastSeenTs[conversationKey] = newest
    }

    override fun onNotificationRemoved(sbn: StatusBarNotification) { /* no-op */ }

    private fun ensureChannel() {
        val mgr = getSystemService(NotificationManager::class.java) ?: return
        if (mgr.getNotificationChannel(CHANNEL_ID) == null) {
            mgr.createNotificationChannel(
                NotificationChannel(CHANNEL_ID, "SIRENBERT scam alerts", NotificationManager.IMPORTANCE_HIGH).apply {
                    description = "Warns when a conversation looks like a pig-butchering / romance / investment scam."
                }
            )
        }
    }

    private fun alert(pkg: String, conversationKey: String, conversation: String, verdict: String, pred: OnDevicePrediction) {
        // PII rule (AGENTS.md): never log message bodies or contact names. Package + verdict +
        // probs + trigger are safe to log; the conversation name is shown only in the user-facing alert.
        Timber.tag(TAG).w(
            "cross-app %s in %s  scam=%.2f susp=%.2f trigger=%s",
            verdict, pkg,
            pred.scamProbability ?: 0f, pred.suspiciousProbability ?: 0f, pred.messageTrigger ?: "?",
        )
        val prob = (if (verdict == "SCAM") pred.scamProbability else pred.suspiciousProbability) ?: 0f
        val title = if (verdict == "SCAM") "Possible scam detected" else "Suspicious conversation"
        val text = "SIRENBERT flagged “$conversation” (${(prob * 100).toInt()}% ${verdict.lowercase()})"
        val notification = NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.stat_sys_warning)
            .setContentTitle(title)
            .setContentText(text)
            .setStyle(NotificationCompat.BigTextStyle().bigText(text))
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setCategory(NotificationCompat.CATEGORY_ALARM)
            .setAutoCancel(true)
            .build()
        try {
            // Stable per-conversation id so repeated alerts on the same chat update rather than stack.
            NotificationManagerCompat.from(this)
                .notify(ALERT_ID_BASE + (conversationKey.hashCode() and 0xFFFF), notification)
        } catch (e: SecurityException) {
            // POST_NOTIFICATIONS not granted (Android 13+); detection still logged above.
            Timber.tag(TAG).w("POST_NOTIFICATIONS not granted; cannot show cross-app alert")
        }
    }

    private companion object {
        const val TAG = "SIRENBERT"
        const val CHANNEL_ID = "sirenbert_scam_alerts"
        const val ALERT_ID_BASE = 47_000
    }
}
