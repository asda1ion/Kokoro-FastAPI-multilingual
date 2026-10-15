# Multilingual speech

The web player's default **Multilingual** option sends `lang_code: "multi"` to `POST /v1/audio/speech`. Lingua classifies individual words offline, then adjacent words with the same language share a pronunciation span. `auto` remains an API alias for the same behavior. Every span keeps the request's selected voice and speed. Audio is encoded and streamed through the existing chunked path; language preparation does not generate or buffer audio.

```json
{
  "input": "Привет, people! Bonjour tout le monde! 日本語です。",
  "voice": "af_bella",
  "lang_code": "multi",
  "speed": 1.0,
  "response_format": "mp3"
}
```

Omitting `lang_code` retains the existing configured language (`DEFAULT_VOICE_CODE`) or voice-prefix behavior. Explicit native language requests bypass detection. A language override changes pronunciation, not the selected voice tensor. Pronunciation quality with a voice trained on another language needs listening checks.

| Native code | ISO / locale aliases | Pronunciation |
|---|---|---|
| `a` | `en`, `en-US` | American English Misaki |
| `b` | `en-GB` | British English Misaki |
| `e` | `es` | Spanish Kokoro eSpeak G2P |
| `f` | `fr`, `fr-FR` | French Kokoro eSpeak G2P |
| `h` | `hi` | Hindi Kokoro eSpeak G2P |
| `i` | `it` | Italian Kokoro eSpeak G2P |
| `p` | `pt`, `pt-BR` | Brazilian Portuguese Kokoro eSpeak G2P |
| `j` | `ja` | Japanese Misaki |
| `z` | `zh`, `zh-CN`, `zh-TW`, `zh-Hans`, `zh-Hant` | Mandarin Chinese Misaki |

Aliases are case-insensitive and accept underscores in place of hyphens. Japanese requires the full UniDic dictionary installed by the Docker images or the native Mac launcher. A native route describes the model's pronunciation support, not which dictionaries are installed on a particular local host.

Regional tags such as `es-ES`, `hi-IN`, `it-IT` and `ja-JP` resolve to the primary native language route. They do not add regional accents beyond the registered native pipelines.

On this macOS deployment, Misaki initializes the bundled eSpeak library and data paths. A missing system `espeak-ng` executable alone does not imply missing G2P support, and a working bundled engine does not require a Homebrew install. Initialization must be checked in the environment used to launch the server.

Automatic detection uses all of Lingua's supported languages, including languages without native Kokoro speech support. `GET /v1/audio/languages` reports `protocol_version: 1`, `native` routes, `auto_detection.languages` (ISO 639-1 codes), installed `experimental` manual routes, and `phoneme_bridge` enabled/available flags. Detection coverage does not promise speech support. Bridged speech is always labeled experimental.

Word scores with a top-two margin below 0.15, or words of five letters or fewer with a leading score below 0.5, use the nearest confident, compatible word in the same Unicode sentence (earlier words win distance ties), then a confident same-script phrase, then the selected voice's language if it is a positive-scoring candidate. Without compatible context or voice, the highest-scoring language wins. Cyrillic words therefore never inherit English pronunciation: `Привет!` uses `espeak:ru`, and `Привет, people!` uses Russian then English. Script changes inside words are split, while contiguous Han/kana text stays together for native CJK pronunciation. Repeated word scores are reused within each request.

Detected English keeps British pronunciation when the fallback voice is British. Unicode is normalized to NFC without translation or letter removal. Custom phoneme units remain atomic. Cyrillic text goes through language detection rather than a script-to-Russian shortcut. Unknown scripts and unidentifiable text return an error suggesting an explicit `lang_code`. Automatic input containing only emoji, punctuation or symbols refuses before audio headers with an actionable HTTP 400. Numbers, nonempty custom IPA and positive pause tags still prepare, and punctuation and emoji within text are retained according to normalization options. Shared words and very short text remain inherently ambiguous; explicit languages bypass those choices.

All spans are checked before the speech response opens. A detected language without an installed pronunciation route returns HTTP 400. Unsupported explicit language codes return a request validation error. The experimental eSpeak bridge described below supplies non-native pronunciation; its kill switches preserve native multilingual routing.

English expansion rules only apply to English spans. Non-English numbers and meaningful letters reach native G2P unchanged. Existing normalization switches, emoji removal, custom phoneme notation, pause tags and opt-in voice/rate tags retain their policies. With voice tags disabled, bracketed text is literal; automatic detection does not grant it control authority.

Routing and error behavior have model-free tests. Real multilingual pronunciation, cross-language voice quality and Companion integration require the separate cross-component validation task.

## Experimental installed eSpeak bridge

Every installed eSpeak language/dialect is listed under `experimental` with an explicit `espeak:<identifier>` manual route. Plain identifiers and locale aliases prefer native Kokoro routes; other installed identifiers resolve to the bridge. For example, `fr` stays native, `ru` becomes `espeak:ru`, and `espeak:fr` explicitly requests the experimental engine. Dialects such as `vi-vn-x-central` retain their installed identifiers. Unknown identifiers fail validation. Manual coverage depends on the installed library/data, independently of Lingua detection coverage; `auto_detection.speech_languages` lists the detector languages with an installed pronunciation route. Unmapped IPA can still reject a particular input on any experimental route.

Non-native spans in `multi` (or its `auto` alias) use this bridge. `PHONEME_BRIDGE=false` or request `phoneme_bridge: false` rejects experimental routes before streaming while preserving native requests. The bundled library and data paths are configured explicitly before discovery, so a macOS environment needs no incidental English warmup or system Homebrew eSpeak. Missing library/data yields an empty experimental inventory and actionable rejection, rather than a silent English fallback.

Preparation uses eSpeak IPA with stress and punctuation, then checks every mapped character against the **loaded KModel's config vocabulary**, not the separate generic text-processing vocabulary. Prepared chunks are bounded by the loaded context length minus BOS/EOS, the pipeline's 510-character limit, and `ABSOLUTE_MAX_TOKENS`. All preparation finishes before response headers; synthesis uses `generate_from_tokens` directly and keeps the selected voice, speed, ordering, pause tags and cancellation. IPA is never submitted to English G2P. Non-native numbers and accents are retained; English normalization is not applied. Custom `[word](/IPA/)` units are also checked against the loaded vocabulary. Experimental chunks have no inferred word captions; chunk timings may associate text with the first piece when a word's IPA needs several model chunks.

`phoneme_bridge.mapping` publishes the mapping version, deterministic SHA-256 and target-vocabulary provenance. Ordinary bridge logs contain route, mapping version, vocabulary hash, chunk count and limit, without input text or IPA. Synthetic validation artifacts deliberately include the corpus and mappings for audit.

### Mapping `espeak-kokoro-ipa-v2`

Version 2 adds explicit `ʑ` to `ʒ` and `ɪ^` to `ʲ` approximations and changes the deterministic mapping hash. The loaded model must contain each replacement; otherwise preparation still refuses. A model that already contains `ʑ` retains it unchanged. eSpeak 1.52 emits `ɪ^` in Russian `проверьте`; its [Russian phoneme source](https://github.com/espeak-ng/espeak-ng/blob/1.52.0/phsource/ph_russian#L185-L196) defines this nonsyllabic “silent i” as palatalizing the preceding consonant. Only that complete sequence maps to the model’s palatalization marker; a bare `^` still refuses.

The executable table is `api/src/services/text_processing/phoneme_bridge.py`. IPA symbols already present in the loaded vocabulary stay unchanged, including primary/secondary stress, length, nasalization and soft consonants (`ʲ`). When absent, soft consonants approximate with a following `j`, length can only survive when `ː` exists, and unknown symbols reject with sanitized Unicode code points and mapping-version guidance.

| Unsupported form | Approximation |
|---|---|
| Tied affricates `d͡ʒ`, `t͡ʃ`, `d͡z`, `t͡s` (also below tie) | `ʤ`, `ʧ`, `ʣ`, `ʦ` |
| Latin `g`, dark/retroflex/lateral `ɫ ɭ ɬ ɮ`, retroflex `ʐ` | `ɡ`, `l l l z`, `ʒ` |
| Alveolo-palatal `ʑ` (U+0291), observed in Russian `друзья` | `ʒ`; palatal articulation contrast lost |
| Russian nonsyllabic `ɪ^`, observed in `проверьте` | `ʲ`; preceding-consonant palatalization approximation |
| Breath/pharyngeal `ɦ ħ ʕ ʜ ʢ` | `h h ʔ h ʔ` |
| Uvular/implosive `ɢ ɠ ɓ ɗ ʄ ʛ` | `ɡ ɡ b d ɟ ɡ` |
| `ɺ ɱ ɧ ɞ ɘ ɵ ɶ ʉ ɿ ʅ` | `ɾ m x ə ə ø œ u ɨ ɨ` |
| Secondary `ʷ ˠ ˤ ˞ ˑ` | `w ɣ ʔ ɹ ː` |
| Syllabic consonant / creaky mark | Preceding `ə` / following `ʔ` |
| Tone levels `˥ ˦ ˧ ˨ ˩`, numeric tones 1–9 | Nearest model arrow level; contours retained in sequence |
| Dental, apical, laminal, release, advanced/retracted, raised/lowered, voicing, short and nonsyllabic marks | Retain base segment; articulation contrast lost, substitution recorded |
| Remaining tie bars | Join adjacent phonemes; no segment removed |
| Hyphen/whitespace and typographic punctuation | Model space or corresponding punctuation |

Every approximation, including articulation-mark simplification, is recorded by preparation; there is no generic unknown-character filter. Replacement characters must themselves exist in the loaded model. This is best-effort pronunciation, not new language training: pharyngeal/voicing contrasts, tones, dialect nuance and cross-language voice quality can be inaccurate. A successful nonempty WAV proves valid synthesis, not intelligibility.

For a bounded local MPS check, run `uv run --no-sync python -m examples.multilingual_bridge_smoke --launch --output .orbit/tmp/mps-smoke` **from the TTS repository root**. The harness starts `./start-gpu_mac.sh` with loopback HOST and an unused explicit PORT, collects synthetic WAV/PCM and routing/mapping/timing evidence, and waits for its own server to stop. It does not touch other listeners, Docker, microphones, speakers or paid providers. Existing native environments install new core requirements while retaining installed extras; fresh environments use the CPU PyTorch wheel with Apple MPS support.

Deep macOS worktree paths can make the bundled eSpeak initializer fall back to an invalid build-time data path and exit the process. The server copies the bundled data into an owned temporary directory, cleans it at process exit, and explicitly restores wrapper paths after importing Misaki and before constructing native or bridge G2P. The exact failing path-length threshold is not established. This is an initialization/path issue, not evidence that the host lacks language data. No Homebrew install or shorter checkout is required.

### Native Mac Japanese setup

Run `HOST=127.0.0.1 PORT=8911 ./start-gpu_mac.sh` from the repository root, choosing an unused port. Before uvicorn starts, the launcher checks full UniDic files and Japanese token pronunciations in its own virtualenv. Fresh and existing environments with missing default data download full UniDic (a roughly 526 MB archive, requiring additional space for extraction); later launches verify and reuse it. Packages alone, including `unidic-lite`, are insufficient.

Downloads are staged and checked before missing files are installed. Existing dictionary files, installed extras, model weights and voices are preserved. The check uses fugashi's supported `unidic.DICDIR` selection; it does not invent a `UNIDIC_DICDIR` environment override. An explicitly configured dictionary must already be usable. Existing corrupt files or an invalid explicit dictionary stop startup with an error rather than being overwritten. Repair that data and rerun the launcher; a download or verification failure never starts the server. The standalone check is `uv run --no-sync python -m api.src.core.japanese`.
