# LangSwap

LangSwap is a free Windows desktop application for turning an English video
into a French MP4 with source and translated SRT subtitles. Paste a YouTube
link or choose a local file, select a voice mode, and choose an output folder.

**Download:** [LangSwap for Windows](https://le-sieur.ca/langswap/).
The ZIP contains the application and its corresponding source code.

## Voice modes

| Mode | Input | Output |
| --- | --- | --- |
| Qwen3-TTS | YouTube video with one narrator | French speech based on the English voice reference |
| OpenVoice | YouTube or local video | French speech with a consistent voice per detected speaker |
| Existing YouTube track | YouTube video with a translated audio track | Video and subtitles in the selected available language |

Voice resemblance, translation, speaker detection and timing should be
reviewed before sharing a result. Use content and voices you have permission
to adapt.

## Windows setup

Use Windows 10 or 11 with .NET Framework 4.8 and Docker Desktop running Linux
containers. Local voice modes need a compatible NVIDIA GPU. The first job
builds Docker images and downloads model weights, which requires Internet
and several gigabytes of free space. An existing YouTube translated track
does not need a GPU.

Extract the ZIP completely and run LangSwap.exe. The app stores models and
jobs in its local application data folder and exports to Videos/LangSwap by
default. Set LANGSWAP_MODELS_DIR, LANGSWAP_JOBS_DIR or LANGSWAP_OUTPUT_DIR
before starting the app to choose different folders.

The executable is currently unsigned. See
[Windows instructions](desktop/README-Windows.md) for details.

## Source and build

- desktop/LangSwap.Desktop/Program.cs — native Windows interface.
- runtime/langswap — Python processing engine and Docker recipes.
- runtime/langswap/tests — processing tests.

On Windows, run desktop/Build-Release.ps1 to compile the executable and create
the source-inclusive ZIP. The script uses the .NET Framework compiler included
with Windows. The generated ZIP, media, model weights and job data are excluded
from Git.

## Licence

This code includes modifications dated 2026-11-04 and is distributed under
AGPL-3.0-or-later. See [NOTICE.md](NOTICE.md) and [LICENSE](LICENSE). External
model weights and dependencies have their own licences; they are downloaded
separately and are not included in this repository.
