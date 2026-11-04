# LangSwap — vidéos anglaises vers français

Ce fork transforme une vidéo anglaise en vidéo française avec transcription,
traduction, voix par narrateur, musique conservée et deux fichiers SRT.
Son interface permet de coller une URL YouTube ou d'importer un fichier,
de choisir les langues et le dossier de sortie, puis de comparer son doublage
à une piste française YouTube existante lorsqu'elle est disponible.
Le profil local par défaut est **Whisper/VAD + ECAPA → Marian → Kokoro FR +
OpenVoice → synchronisation sur les temps de la vidéo**.

Les identités vocales sont calculées sur toute la vidéo. Une personne conserve
sa référence lorsqu'elle revient après une autre personne. Le timbre obtenu
reste une approximation : vérifier à l'écoute la prononciation, les noms propres,
la traduction et la ressemblance vocale.

## Installer le profil français

Linux, Python **3.11 ou 3.12**, GPU NVIDIA/CUDA 12.4 pour l'accélération.
Installer `ffmpeg`, `rubberband-cli`, `espeak-ng`, `libsndfile1` et Git.
Depuis ce dossier `runtime/langswap` :

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[french,dev]"
uv pip install --no-deps "MyShell-OpenVoice @ git+https://github.com/myshell-ai/OpenVoice.git@74a1d147b17a8c3092dd5430504bd83ef6c7eb23"
python main.py local in.mp4 french english
```

La seconde installation est intentionnelle : les anciennes contraintes de
MyShell-OpenVoice ne décrivent pas les versions compatibles utilisées pour son
convertisseur. L'extra `french` fournit ces dépendances ; `--no-deps` empêche
OpenVoice de les rétrograder. Les profils `gpu` historiques pour OmniVoice et
`api` sont des environnements distincts, pas des compléments de `french`.

Les poids se téléchargent au premier usage dans `MODEL_WEIGHTS_DIR` ; garder ce
dossier hors Git. Les données de jobs se trouvent dans `LANGSWAP_DATA_DIR`.
Le profil français local n'utilise aucune clé d'API et n'envoie pas l'audio au cloud.

## Image dédiée

```bash
docker build -f Dockerfile.french -t langswap-french .
docker run --rm --gpus all \
  -v "$PWD/models_weights:/app/models_weights" \
  -v "$PWD/data:/app/data" \
  -v "$PWD/input:/input:ro" \
  langswap-french local /input/in.mp4 french english
```

Cette image sert le CLI et l'interface locale. Elle est indépendante des anciennes images OmniVoice,
XTTS et Chatterbox. Le fichier source fourni reste accessible en lecture seule.
Le résultat est `data/<id>/resulted_video.mp4`, avec
`source_transcript.srt` et `translated_transcript.srt`.

### Application de bureau Windows

La distribution gratuite contient LangSwap.exe et ce moteur dans runtime/langswap.
Décompresser le ZIP entier et ouvrir l'exécutable sur Windows 10/11 avec
.NET Framework 4.8 et Docker Desktop démarré. L'application accepte une URL
YouTube ou un fichier local, les langues et un dossier de sortie.

- **Voix proche · Qwen** : une voix française issue de la référence anglaise
  d'un narrateur YouTube. Nécessite un GPU NVIDIA compatible.
- **Voix locale · OpenVoice** : plusieurs locuteurs, avec une référence de voix
  réutilisée pour chaque personne. Accepte aussi les vidéos locales.
- **Piste traduite de YouTube** : récupère une piste traduite déjà disponible,
  sans génération vocale et sans GPU.

Le premier lancement d'un traitement construit les images Docker et télécharge
les poids nécessaires. Les modèles, jobs et vidéos restent dans des dossiers
locaux. Aucun compte ni clé API n'est requis. L'URL YouTube vise la vidéo
entière : les paramètres de temps et de liste sont ignorés.

L'installation Windows est détaillée dans
[README-Windows.md](../../desktop/README-Windows.md).
gradio_demo.py reste une interface de développement locale.
Pour l'utiliser sur Docker, monter les
dossiers des poids, des jobs et de la sortie sur /app/models_weights,
/app/data et /app/output, publier le port 7860 sur 127.0.0.1 et
lancer gradio_demo.py --host 0.0.0.0 dans l'image.

## Voix et corrections manuelles

- **OpenVoice** : prononciation française de Kokoro, référence distincte et stable
  pour chaque narrateur. Une référence manquante ou une erreur bloque le job ;
  aucun autre narrateur ni silence n'est substitué en cachette.
- **Kokoro** : voix française fixe, sans clonage. Ce choix n'adapte pas
  automatiquement une voix masculine ou féminine à chaque personne.
- **Edge** : service en ligne, voix de substitution choisies explicitement par
  personne. Par exemple, après inspection des identités :

```bash
python main.py local in.mp4 french english --stop-after asr
python main.py local in.mp4 french english --tts edge \
  --speaker-voice SPEAKER_00=fr-CA-JeanNeural \
  --speaker-voice SPEAKER_01=fr-CA-SylvieNeural
```

Edge nécessite `edge-tts` dans un environnement où ce backend est installé,
ainsi qu'Internet. Le profil local ne bascule pas automatiquement vers ce service.
Choisir des voix adaptées aux personnes ; le logiciel ne déduit pas leur sexe.
Pour une vidéo confirmée à une seule voix : `--no-diarization --speakers 1`.
Un nombre connu de narrateurs peut être fourni avec `--speakers N`.

L'éditeur recharge les références et invalide les caches lors d'un changement de
texte, de timing ou de narrateur. Les réécritures de vidéo sont atomiques et
conservent le résultat précédent si FFmpeg échoue. Des fenêtres exactement
identiques pour plusieurs segments sont rejetées afin d'éviter l'écrasement de
leurs fichiers audio.

## Vérifier le code

```bash
pytest -q
ruff check --isolated --select E9,F63,F7,F82 .
```

Les tests CPU couvrent transcription/identités, caches, fichiers, traduction
complète, voix, édition, timing, FFmpeg et fragments HLS.

## Licence et modifications

Ce code comporte des modifications datées du 4 novembre 2026. Il est
distribué sous **AGPL-3.0-or-later**, dont le texte se trouve dans
[LICENSE](LICENSE). Les poids et bibliothèques tierces conservent leurs
propres licences.
