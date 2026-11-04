# LangSwap pour Windows

LangSwap est une application de bureau gratuite. Elle prend une URL YouTube ou
un fichier vidéo, puis enregistre une vidéo doublée et deux fichiers SRT dans
le dossier choisi. Le programme ne demande ni compte ni clé API.

## Installation

1. Utiliser Windows 10 ou 11 avec .NET Framework 4.8.
2. Installer et démarrer Docker Desktop en mode conteneurs Linux.
3. Décompresser **tout** le ZIP, puis ouvrir LangSwap.exe.
4. Coller une URL YouTube HTTPS ou sélectionner une vidéo locale, choisir le
   français et un dossier de sortie, puis lancer le traitement.

Le premier traitement construit les images Docker et télécharge les modèles.
Prévoir plusieurs gigaoctets libres et une connexion Internet. Les modes de
voix locale demandent une carte NVIDIA compatible avec Docker GPU/CUDA.
L'exécutable n'est pas encore signé numériquement.
Le mode « Piste traduite de YouTube » peut fonctionner sans GPU si la vidéo
offre déjà la piste audio dans la langue choisie.

La voix Qwen est destinée à une vidéo YouTube avec **un narrateur**. Le profil
OpenVoice gère plusieurs locuteurs et accepte aussi un fichier local. Si une
piste française automatique de YouTube existe, elle peut servir de comparaison.
Les langues autres que le français concernent seulement cette dernière piste.

Le dossier runtime/langswap contient le code Python et les recettes Docker.
desktop/LangSwap.Desktop/Program.cs est le code source de l'interface Windows.
Le moteur local et les fichiers de travail restent sur votre ordinateur.
Les téléchargements depuis YouTube et des modèles requièrent Internet.
Par défaut, les modèles et jobs sont dans le dossier local de l'application
et les exports dans Vidéos/LangSwap. Pour choisir d'autres emplacements avant
de lancer le programme, définir LANGSWAP_MODELS_DIR, LANGSWAP_JOBS_DIR ou
LANGSWAP_OUTPUT_DIR dans l'environnement Windows.

Relisez et écoutez les noms, chiffres, voix et sous-titres avant de diffuser.
N'utilisez que des contenus et voix que vous avez le droit d'adapter.

Code sous AGPL-3.0-or-later. Les modèles et dépendances ont leurs propres
licences. La copie de la licence se trouve dans LICENSE.
