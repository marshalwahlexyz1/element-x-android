/*
 * Copyright (c) 2025 Element Creations Ltd.
 *
 * SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Element-Commercial.
 */

package io.element.android.features.messages.impl.sirenbert

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import timber.log.Timber

/**
 * Foreground service that runs the full bulk on-device replay (all test
 * conversations) without Android suspending it.
 *
 * The bulk replay over the whole held-out test set takes tens of minutes of
 * sustained ORT inference. As a plain background coroutine it was getting
 * throttled/killed when the app left the foreground. A foreground service with
 * an ongoing notification keeps the process alive and CPU-scheduled for the
 * duration, so the run completes reliably and uses the full test set (no
 * subsetting needed for the equivalence/accuracy/latency numbers).
 *
 * Started from [SirenbertResetReceiver] on the SIRENBERT_BULK_REPLAY broadcast.
 * The single-run guard lives in [SirenbertBulkReplay]; a second start while a
 * run is in flight is ignored there.
 */
class SirenbertBulkReplayService : Service() {

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        startForeground(NOTIF_ID, buildNotification("Running on-device SIRENBERT replay…"))
        scope.launch {
            try {
                SirenbertBulkReplay.runOnce(applicationContext)
            } finally {
                stopForegroundCompat()
                stopSelf()
            }
        }
        return START_NOT_STICKY
    }

    private fun buildNotification(text: String): Notification {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val mgr = getSystemService(NotificationManager::class.java)
            mgr.createNotificationChannel(
                NotificationChannel(CHANNEL_ID, "SIRENBERT evaluation", NotificationManager.IMPORTANCE_LOW)
            )
        }
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("SIRENBERT bulk replay")
            .setContentText(text)
            .setSmallIcon(android.R.drawable.stat_sys_download)
            .setOngoing(true)
            .build()
    }

    @Suppress("DEPRECATION")
    private fun stopForegroundCompat() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            stopForeground(STOP_FOREGROUND_REMOVE)
        } else {
            stopForeground(true)
        }
    }

    companion object {
        private const val CHANNEL_ID = "sirenbert_eval"
        private const val NOTIF_ID = 4242

        fun start(context: Context) {
            val intent = Intent(context, SirenbertBulkReplayService::class.java)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                context.startForegroundService(intent)
            } else {
                context.startService(intent)
            }
            Timber.tag("SIRENBERT").i("bulk replay foreground service started")
        }
    }
}
