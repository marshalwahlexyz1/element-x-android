/*
 * Copyright (c) 2025 Element Creations Ltd.
 *
 * SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Element-Commercial.
 */

package io.element.android.features.messages.impl.sirenbert

import io.element.android.features.messages.impl.BuildConfig
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import okhttp3.Call
import okhttp3.Callback
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import timber.log.Timber
import java.io.IOException
import java.util.concurrent.TimeUnit
import kotlin.coroutines.resume
import kotlinx.coroutines.suspendCancellableCoroutine

/**
 * Minimal Phase A.2 HTTP client for the laptop FastAPI SIRENBERT server.
 *
 * Endpoint, request and response shapes mirror what the laptop server already
 * implements (see redefined_approach/docs/12_ELEMENT_X_FASTAPI_DEMO.md):
 *
 *   POST {SIRENBERT_API_URL}/predict
 *   { conversation_id, message_id, role: "S"|"T", message }
 *
 * Suspect ("S") responses populate trigger / state / probs. Target ("T")
 * responses come back as CONTEXT_ONLY.
 *
 * Phase A.2 is intentionally minimal:
 *   - one shared OkHttp client (singleton)
 *   - kotlinx.serialization for body marshalling
 *   - 8s timeouts
 *   - the caller (SirenbertCache) is responsible for not double-firing
 */
internal object SirenbertApiClient {
    private val json = Json {
        ignoreUnknownKeys = true
        coerceInputValues = true
    }

    private val mediaType = "application/json; charset=utf-8".toMediaType()

    private val client: OkHttpClient by lazy {
        OkHttpClient.Builder()
            .connectTimeout(8, TimeUnit.SECONDS)
            .readTimeout(8, TimeUnit.SECONDS)
            .writeTimeout(8, TimeUnit.SECONDS)
            .retryOnConnectionFailure(false)
            .build()
    }

    @Serializable
    private data class PredictRequest(
        val conversation_id: String,
        val message_id: String,
        val role: String,
        val message: String,
    )

    @Serializable
    internal data class PredictResponse(
        val verdict: String? = null,
        @SerialName("message_trigger") val messageTrigger: String? = null,
        @SerialName("fsm_state") val fsmState: String? = null,
        @SerialName("suspicious_probability") val suspiciousProbability: Float? = null,
        @SerialName("conversation_scam_probability") val scamProbability: Float? = null,
        val error: String? = null,
    )

    /**
     * Suspending POST /predict. Returns the parsed response on success, or
     * throws IOException on any network / parse failure. The caller maps that
     * onto a [SirenbertResult] with status Error.
     */
    suspend fun predict(
        roomId: String,
        eventId: String,
        role: String,
        body: String,
    ): PredictResponse {
        val url = "${BuildConfig.SIRENBERT_API_URL.trimEnd('/')}/predict"
        val requestBody = json.encodeToString(
            PredictRequest.serializer(),
            PredictRequest(
                conversation_id = roomId,
                message_id = eventId,
                role = role,
                message = body,
            ),
        )
        val httpRequest = Request.Builder()
            .url(url)
            .post(requestBody.toRequestBody(mediaType))
            .build()
        Timber.tag("SIRENBERT").d(
            "POST /predict role=%s room=%s id=%s url=%s",
            role,
            roomId,
            eventId.take(12),
            url,
        )
        return suspendCancellableCoroutine { cont ->
            val call = client.newCall(httpRequest)
            cont.invokeOnCancellation { call.cancel() }
            call.enqueue(object : Callback {
                override fun onFailure(call: Call, e: IOException) {
                    Timber.tag("SIRENBERT").w(
                        e,
                        "POST /predict FAILED id=%s",
                        eventId.take(12),
                    )
                    if (cont.isActive) cont.resumeWith(Result.failure(e))
                }

                override fun onResponse(call: Call, response: Response) {
                    response.use {
                        val raw = it.body?.string().orEmpty()
                        if (!it.isSuccessful) {
                            Timber.tag("SIRENBERT").w(
                                "POST /predict HTTP %d id=%s",
                                it.code,
                                eventId.take(12),
                            )
                            if (cont.isActive) {
                                cont.resumeWith(Result.failure(IOException("HTTP ${it.code}")))
                            }
                            return@use
                        }
                        try {
                            val parsed = json.decodeFromString(
                                PredictResponse.serializer(),
                                raw,
                            )
                            if (cont.isActive) cont.resume(parsed)
                        } catch (t: Throwable) {
                            Timber.tag("SIRENBERT").w(t, "POST /predict parse error")
                            if (cont.isActive) cont.resumeWith(Result.failure(IOException(t)))
                        }
                    }
                }
            })
        }
    }
}
