/*
 * Copyright (c) 2025 Element Creations Ltd.
 *
 * SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Element-Commercial.
 */

package io.element.android.features.messages.impl.sirenbert

import java.text.Normalizer

/**
 * Minimal WordPiece tokenizer compatible with `distilbert-base-uncased`.
 *
 * Mirrors the behaviour of HuggingFace `DistilBertTokenizer` /
 * `BertTokenizer(do_lower_case=True)` closely enough that token ids match
 * end-to-end on plain English text. We intentionally do NOT reimplement
 * unicode segmentation rules in full; the test corpus is English, the
 * server uses the same vocab, and the integration test (replay-and-diff)
 * will tell us if we ever diverge.
 *
 * Loaded from the `assets/sirenbert/vocab.txt` shipped alongside the ONNX
 * model artifacts. See [OnDeviceSirenbertEngine] for the full pipeline.
 */
class BertWordPieceTokenizer(
    private val vocab: Map<String, Int>,
    private val doLowerCase: Boolean = true,
    private val unkToken: String = "[UNK]",
    private val sepToken: String = "[SEP]",
    private val clsToken: String = "[CLS]",
    private val padToken: String = "[PAD]",
    private val maxInputCharsPerWord: Int = 200,
) {
    val unkId: Int = vocab.getValue(unkToken)
    val sepId: Int = vocab.getValue(sepToken)
    val clsId: Int = vocab.getValue(clsToken)
    val padId: Int = vocab.getValue(padToken)

    /**
     * Encode an ordered list of text segments as a single BERT-style input
     * `[CLS] seg0 [SEP] seg1 [SEP] ... segN [SEP]`. Pads/truncates to
     * [maxLength]. Returns the resulting id sequence and attention mask.
     *
     * Caller is responsible for choosing [maxLength] (typically 512 for
     * DistilBERT).
     */
    fun encodeSegments(segments: List<String>, maxLength: Int): TokenizedInput {
        val ids = ArrayList<Int>(maxLength)
        ids.add(clsId)
        for (segment in segments) {
            for (token in tokenize(segment)) {
                ids.add(vocab[token] ?: unkId)
                if (ids.size >= maxLength - 1) break
            }
            ids.add(sepId)
            if (ids.size >= maxLength) break
        }
        // Truncate to maxLength (the inner loop is conservative, this is a
        // safety net for very long single tokens or weird inputs).
        val truncated = if (ids.size > maxLength) ids.subList(0, maxLength) else ids
        val realLen = truncated.size
        val padded = LongArray(maxLength)
        val mask = LongArray(maxLength)
        for (i in 0 until realLen) {
            padded[i] = truncated[i].toLong()
            mask[i] = 1L
        }
        for (i in realLen until maxLength) {
            padded[i] = padId.toLong()
            mask[i] = 0L
        }
        return TokenizedInput(padded, mask)
    }

    /**
     * Lower-level: full tokenization of one text segment into WordPiece
     * subword strings (NOT yet ids).
     */
    internal fun tokenize(text: String): List<String> {
        val normalized = basicTokenize(text)
        val result = ArrayList<String>(normalized.size * 2)
        for (token in normalized) {
            result.addAll(wordpieceTokenize(token))
        }
        return result
    }

    private fun basicTokenize(text: String): List<String> {
        // 1. Optionally lowercase.
        val lowered = if (doLowerCase) text.lowercase() else text
        // 2. Strip accents (NFKD + drop combining marks).
        val nfd = Normalizer.normalize(lowered, Normalizer.Form.NFD)
        val sb = StringBuilder(nfd.length)
        for (c in nfd) {
            if (Character.getType(c).toByte() != Character.NON_SPACING_MARK) {
                sb.append(c)
            }
        }
        val stripped = sb.toString()
        // 3. Split on whitespace, then split each chunk on punctuation
        //    boundaries (each punct char becomes its own token).
        val tokens = ArrayList<String>(stripped.length / 4 + 1)
        for (chunk in stripped.split(WHITESPACE)) {
            if (chunk.isEmpty()) continue
            val cur = StringBuilder()
            for (ch in chunk) {
                if (isPunctuation(ch)) {
                    if (cur.isNotEmpty()) {
                        tokens.add(cur.toString())
                        cur.clear()
                    }
                    tokens.add(ch.toString())
                } else {
                    cur.append(ch)
                }
            }
            if (cur.isNotEmpty()) tokens.add(cur.toString())
        }
        return tokens
    }

    private fun wordpieceTokenize(token: String): List<String> {
        if (token.length > maxInputCharsPerWord) {
            return listOf(unkToken)
        }
        val subs = ArrayList<String>(4)
        var start = 0
        while (start < token.length) {
            var end = token.length
            var found: String? = null
            while (end > start) {
                val piece = (if (start > 0) "##" else "") + token.substring(start, end)
                if (vocab.containsKey(piece)) {
                    found = piece
                    break
                }
                end -= 1
            }
            if (found == null) {
                return listOf(unkToken)
            }
            subs.add(found)
            start = end
        }
        return subs
    }

    private fun isPunctuation(ch: Char): Boolean {
        val code = ch.code
        // ASCII punctuation per the canonical BERT BasicTokenizer.
        if (code in 33..47) return true
        if (code in 58..64) return true
        if (code in 91..96) return true
        if (code in 123..126) return true
        // Unicode general categories Pd, Ps, Pe, Pi, Pf, Pc, Po, etc.
        return when (Character.getType(ch).toByte()) {
            Character.CONNECTOR_PUNCTUATION,
            Character.DASH_PUNCTUATION,
            Character.START_PUNCTUATION,
            Character.END_PUNCTUATION,
            Character.INITIAL_QUOTE_PUNCTUATION,
            Character.FINAL_QUOTE_PUNCTUATION,
            Character.OTHER_PUNCTUATION -> true
            else -> false
        }
    }

    companion object {
        private val WHITESPACE = Regex("\\s+")

        /**
         * Read a `vocab.txt` (one token per line, line index = token id) into
         * a (token -> id) map.
         */
        fun loadVocab(lines: Sequence<String>): Map<String, Int> {
            val m = HashMap<String, Int>(31000)
            for ((i, line) in lines.withIndex()) {
                m[line.trim()] = i
            }
            return m
        }
    }
}

/** Tokenized input ready for an ONNX session that expects int64 tensors. */
data class TokenizedInput(
    val inputIds: LongArray,
    val attentionMask: LongArray,
)
