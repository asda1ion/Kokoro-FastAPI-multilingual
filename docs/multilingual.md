# Multilingual speech

`POST /v1/audio/speech` accepts `lang_code: "auto"` to detect contiguous language spans offline with Lingua. Every span keeps the request's selected voice and speed. Audio is encoded and streamed through the existing chunked path; language preparation does not generate or buffer audio.

```json
{
  "input": "Hello everyone! Bonjour tout le monde! 日本語です。",
  "voice": "af_bella",
  "lang_code": "auto",
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

Aliases are case-insensitive and accept underscores in place of hyphens. Japanese still requires the full UniDic dictionary installed by the Docker images. A native route describes the model's pronunciation support, not which dictionaries are installed on a particular local host.

Regional tags such as `es-ES`, `hi-IN`, `it-IT` and `ja-JP` resolve to the primary native language route. They do not add regional accents beyond the registered native pipelines.

On this macOS deployment, Misaki initializes the bundled eSpeak library and data paths. A missing system `espeak-ng` executable alone does not imply missing G2P support, and a working bundled engine does not require a Homebrew install. Initialization must be checked in the environment used to launch the server.

Automatic detection uses all of Lingua's supported languages, including languages without native Kokoro speech support. `GET /v1/audio/languages` reports `protocol_version: 1`, `native` routes, `auto_detection.languages` (ISO 639-1 codes), an empty `experimental` list, and `phoneme_bridge` enabled/available flags. Detection coverage does not promise speech support. Later experimental routes can extend this response without labeling bridged speech as native.

Short spans with fewer than 13 letters and a small confidence margin inherit the nearest confidently detected phrase (earlier phrase wins ties), then the selected voice's native language. Detected English keeps British pronunciation when the fallback voice is British. Unicode is normalized to NFC without translation or letter removal. Cyrillic text goes through language detection rather than a script-to-Russian shortcut. Unknown scripts and unidentifiable text return an error suggesting an explicit `lang_code`.

All spans are checked before the speech response opens. A detected language without a native route returns HTTP 400 with its language code and a bridge availability message. Unsupported explicit language codes return a request validation error. Experimental bridging is not implemented here: `phoneme_bridge: true` does not enable unsupported speech. Set `PHONEME_BRIDGE=false` globally or `phoneme_bridge: false` per request to prohibit future experimental routes; native multilingual routing remains available.

English expansion rules only apply to English spans. Non-English numbers and meaningful letters reach native G2P unchanged. Existing normalization switches, emoji removal, custom phoneme notation, pause tags and opt-in voice/rate tags retain their policies. With voice tags disabled, bracketed text is literal; automatic detection does not grant it control authority.

Routing and error behavior have model-free tests. Real multilingual pronunciation, cross-language voice quality and Companion integration require the separate cross-component validation task.
