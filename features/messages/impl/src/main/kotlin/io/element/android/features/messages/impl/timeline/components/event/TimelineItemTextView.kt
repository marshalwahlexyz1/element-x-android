/*
 * Copyright (c) 2025 Element Creations Ltd.
 * Copyright 2023-2025 New Vector Ltd.
 *
 * SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Element-Commercial.
 * Please see LICENSE files in the repository root for full details.
 */

package io.element.android.features.messages.impl.timeline.components.event

import android.text.SpannedString
import androidx.annotation.VisibleForTesting
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.height
import androidx.compose.material3.LocalContentColor
import androidx.compose.material3.LocalTextStyle
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.tooling.preview.PreviewParameter
import androidx.compose.ui.unit.dp
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.ui.platform.LocalContext
import io.element.android.compound.theme.ElementTheme
import io.element.android.features.messages.impl.sirenbert.LocalSirenbertOnDevice
import io.element.android.features.messages.impl.sirenbert.LocalSirenbertRoomId
import io.element.android.features.messages.impl.sirenbert.SirenbertCache
import io.element.android.features.messages.impl.sirenbert.SirenbertResult
import io.element.android.features.messages.impl.timeline.components.layout.ContentAvoidingLayout
import io.element.android.features.messages.impl.timeline.components.layout.ContentAvoidingLayoutData
import io.element.android.features.messages.impl.timeline.model.event.TimelineItemTextBasedContent
import io.element.android.features.messages.impl.timeline.model.event.TimelineItemTextBasedContentProvider
import io.element.android.features.messages.impl.timeline.model.event.aTimelineItemTextContent
import io.element.android.features.messages.impl.utils.containsOnlyEmojis
import io.element.android.libraries.androidutils.text.LinkifyHelper
import io.element.android.libraries.designsystem.preview.ElementPreview
import io.element.android.libraries.designsystem.preview.PreviewsDayNight
import io.element.android.libraries.designsystem.theme.components.Text
import io.element.android.libraries.textcomposer.ElementRichTextEditorStyle
import io.element.android.libraries.textcomposer.mentions.LocalMentionSpanUpdater
import io.element.android.wysiwyg.compose.EditorStyledText
import io.element.android.wysiwyg.link.Link
import timber.log.Timber

@Composable
fun TimelineItemTextView(
    content: TimelineItemTextBasedContent,
    onLinkClick: (Link) -> Unit,
    onLinkLongClick: (Link) -> Unit,
    modifier: Modifier = Modifier,
    showSirenbert: Boolean = false,
    sirenbertRole: String = "S",
    sirenbertMessageId: String = "",
    onContentLayoutChange: (ContentAvoidingLayoutData) -> Unit = {},
) {
    val emojiOnly = content.formattedBody.toString() == content.body &&
        content.body.replace(" ", "").containsOnlyEmojis()
    val textStyle = when {
        emojiOnly -> ElementTheme.typography.fontHeadingXlRegular
        else -> ElementTheme.typography.fontBodyLgRegular
    }
    CompositionLocalProvider(
        LocalContentColor provides ElementTheme.colors.textPrimary,
        LocalTextStyle provides textStyle
    ) {
        val text = getTextWithResolvedMentions(content)
        Box(modifier.semantics { contentDescription = content.plainText }) {
            if (showSirenbert) {
                Column {
                    EditorStyledText(
                        text = text,
                        onLinkClickedListener = onLinkClick,
                        onLinkLongClickedListener = onLinkLongClick,
                        style = ElementRichTextEditorStyle.textStyle(),
                        onTextLayout = ContentAvoidingLayout.measureLegacyLastTextLine(onContentLayoutChange = onContentLayoutChange),
                        releaseOnDetach = false,
                    )
                    Spacer(modifier = Modifier.height(4.dp))
                    SirenbertPlaceholderBadge(
                        role = sirenbertRole,
                        messageId = sirenbertMessageId,
                        body = content.body,
                    )
                }
            } else {
                EditorStyledText(
                    text = text,
                    onLinkClickedListener = onLinkClick,
                    onLinkLongClickedListener = onLinkLongClick,
                    style = ElementRichTextEditorStyle.textStyle(),
                    onTextLayout = ContentAvoidingLayout.measureLegacyLastTextLine(onContentLayoutChange = onContentLayoutChange),
                    releaseOnDetach = false,
                )
            }
        }
    }
}

@Composable
private fun SirenbertPlaceholderBadge(
    role: String,
    messageId: String,
    body: String,
) {
    // Phase A.2: badge reads the per-event result from SirenbertCache and
    // triggers a /predict POST via SirenbertCache.ensureClassified on first
    // composition with a non-empty event id.
    //
    // Empty messageId means the event has no durable Matrix event id yet
    // (local-echo). We render '(local)' and skip the API call; the next
    // recomposition (when the eventId resolves) will fire the request.
    val roomId = LocalSirenbertRoomId.current
    val onDevice = LocalSirenbertOnDevice.current
    val ctx = LocalContext.current
    val cacheKey = messageId
    val resultFlow = remember(cacheKey) { SirenbertCache.observe(cacheKey) }
    val result by resultFlow.collectAsState(initial = SirenbertCache.peek(cacheKey))
    LaunchedEffect(roomId, cacheKey, role, onDevice) {
        if (cacheKey.isNotEmpty() && roomId.isNotEmpty()) {
            SirenbertCache.ensureClassified(
                roomId = roomId,
                eventId = cacheKey,
                role = role,
                body = body,
                useOnDevice = onDevice,
                appContext = ctx,
            )
        }
    }
    val shortId = if (messageId.isEmpty()) "(local)" else messageId.take(10)
    val tail = when (result?.status) {
        null, SirenbertResult.Status.Idle -> "…"
        SirenbertResult.Status.InFlight -> "… (req)"
        SirenbertResult.Status.ContextOnly -> "CONTEXT_ONLY"
        SirenbertResult.Status.Classified -> formatClassified(result!!)
        SirenbertResult.Status.Error -> "! ${result?.errorMessage.orEmpty()}"
    }
    Timber.tag("SIRENBERT").d(
        "badge render role=%s id=%s tail=%s",
        role,
        shortId,
        tail,
    )
    Text(
        text = "SIRENBERT[$role] $shortId · $tail",
        style = ElementTheme.typography.fontBodySmRegular,
        color = ElementTheme.colors.textSecondary,
    )
}

private fun formatClassified(r: SirenbertResult): String {
    val pieces = mutableListOf<String>()
    r.verdict?.takeIf { it.isNotBlank() }?.let { pieces += it }
    r.trigger?.let { pieces += it }
    r.state?.let { pieces += it }
    r.suspProb?.let { pieces += "susp=%.2f".format(it) }
    r.scamProb?.let { pieces += "scam=%.2f".format(it) }
    return pieces.joinToString(" · ").ifEmpty { "classified" }
}

@VisibleForTesting(otherwise = VisibleForTesting.PRIVATE)
@Composable
internal fun getTextWithResolvedMentions(content: TimelineItemTextBasedContent): CharSequence {
    val mentionSpanUpdater = LocalMentionSpanUpdater.current
    val bodyWithResolvedMentions = mentionSpanUpdater.rememberMentionSpans(content.formattedBody)
    return SpannedString.valueOf(bodyWithResolvedMentions)
}

@PreviewsDayNight
@Composable
internal fun TimelineItemTextViewPreview(
    @PreviewParameter(TimelineItemTextBasedContentProvider::class) content: TimelineItemTextBasedContent
) = ElementPreview {
    TimelineItemTextView(
        content = content,
        onLinkClick = {},
        onLinkLongClick = {},
    )
}

@Preview
@Composable
internal fun TimelineItemTextViewWithLinkifiedUrlPreview() = ElementPreview {
    val content = aTimelineItemTextContent(
        formattedBody = LinkifyHelper.linkify("The link should end after the first '?' (url: github.com/element-hq/element-x-android/README?)?.")
    )
    TimelineItemTextView(
        content = content,
        onLinkClick = {},
        onLinkLongClick = {},
    )
}

@Preview
@Composable
internal fun TimelineItemTextViewWithLinkifiedUrlAndNestedParenthesisPreview() = ElementPreview {
    val content = aTimelineItemTextContent(
        formattedBody = LinkifyHelper.linkify("The link should end after the '(ME)' ((url: github.com/element-hq/element-x-android/READ(ME)))!")
    )
    TimelineItemTextView(
        content = content,
        onLinkClick = {},
        onLinkLongClick = {},
    )
}
