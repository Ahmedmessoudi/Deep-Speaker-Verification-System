# 📂 Chapitre 7 : Gestion de la Mémoire et Passage à l'Échelle

Ce chapitre décrit les choix d'ingénierie logicielle implémentés dans nos notebooks PyTorch pour permettre un entraînement stable à grande échelle (1 111 locuteurs, 44 Go de données brutes) sous des contraintes matérielles strictes de RAM et de VRAM (Kaggle GPU P100/T4).

---

## 1. Problématique de la Volumétrie de Données

L'entraînement de modèles de traitement de la parole pose des défis de stockage et de mémoire vive importants :
*   Le jeu de données **VoxCeleb1** représente environ **33 Go** de fichiers compressés.
*   Le jeu de données **MUSAN** de bruit de fond pèse environ **11 Go**.
*   **Total combiné** : **44 Go** de données audio brutes sous forme de fichiers `.wav` et `.flac`.

Les machines virtuelles Kaggle disposent de 30 Go de RAM système (CPU) et de 16 Go de VRAM (GPU T4 ou P100). Charger l'ensemble de ces jeux de données en mémoire vive au démarrage du script provoquerait un crash immédiat pour dépassement de mémoire (**OOM - Out of Memory**). Une architecture de chargement de données optimisée est donc indispensable.

---

## 2. Architecture Clé : Le Chargement à la Demande (Lazy Loading)

Le pipeline PyTorch implémenté repose sur le patron de conception (design pattern) du **Lazy Loading**. Au lieu de charger les formes d'onde en mémoire, le chargeur ne manipule que des métadonnées légères.

```mermaid
graph TD
    subgraph CPU RAM (Très légère < 100 Mo)
        A[Liste de Chemins de Fichiers .wav/.flac] --> B[PyTorch Dataset]
    end
    subgraph Processus d'un Batch (Étape par Étape)
        B --> C[Dataloader : Demande Batch]
        C --> D[CPU Worker : Charger Audio depuis le Disque]
        D --> E[CPU Worker : Appliquer Augmentation et Log-Mel]
        E --> F[CPU Worker : Créer Tenseur de Batch]
        F --> G[GPU VRAM : Transférer Batch pour Entraînement]
        G --> H[Garbage Collector : Supprimer l'Audio Brut de la RAM]
    end
```

### Étape 1 : Allocation CPU RAM Proche de Zéro
Lors de l'instanciation de la classe `VoxCelebDataset`, le constructeur scanne le disque et stocke uniquement la liste des chemins de fichiers (chaînes de caractères) et les étiquettes correspondantes (entiers) :
```python
# Stockage en RAM des chaînes de caractères uniquement
self.pairs = [("/kaggle/input/voxceleb/id10001/audio1.wav", "id10001"), ...]
```
Pour plus de 140 000 fichiers audio, le stockage de ces chemins en mémoire vive CPU consomme **moins de 2 Mo**, ce qui est totalement négligeable.

### Étape 2 : Chargement Temporel à la Milliseconde (On-the-Fly)
C'est uniquement lors de la création d'un lot d'entraînement (Batch) par le `DataLoader` que la méthode `__getitem__(idx)` est déclenchée :
1.  **Lecture Disque** : Le fichier audio ciblé par l'index est lu depuis le disque dur SSD de Kaggle via la bibliothèque `librosa.load(path, sr=16000)`. Seule la portion de 3 secondes nécessaire est extraite et décodée en tableau NumPy.
2.  **Augmentation acoustique** : Si l'augmentation est activée (60% de chances), un fichier de bruit MUSAN est lu à son tour depuis le disque, décodé, mis à l'échelle selon le SNR cible, puis sommé à l'audio propre.
3.  **Extraction de Caractéristiques** : L'onde temporelle mixte de 3 secondes est convertie en spectrogramme Mel, puis en logarithme Log-Mel de dimension $[80 \times 300]$ (80 filtres Mel, 300 trames temporelles de 10 ms).
4.  **Libération Immédiate** : Dès que le tenseur Log-Mel est généré, les signaux temporels bruts NumPy (qui pèsent lourd en mémoire) ne sont plus référencés. Le ramasse-miettes (Garbage Collector) de Python les élimine immédiatement de la RAM système CPU.

---

## 3. Pré-chargement Multi-Processus (Multi-Process Pre-Fetching)

Charger les fichiers audio depuis le disque dur en cours d'entraînement peut créer un goulot d'étranglement (I/O Bottleneck), forçant le GPU à attendre que le CPU finisse de décoder les fichiers audio. Pour éviter cela, nous configurons le `DataLoader` avec `NUM_WORKERS = 4` :

*   PyTorch crée **4 sous-processus CPU indépendants** qui s'exécutent en parallèle du processus d'entraînement principal.
*   Pendant que le GPU calcule les gradients sur le lot actuel (Batch $N$), les workers CPU chargent, augmentent et convertissent déjà les audios pour les lots suivants (Batch $N+1$ à $N+4$) en arrière-plan.
*   Cette stratégie masque entièrement le temps de lecture disque (I/O) sans augmenter la mémoire vive requise de façon permanente.

---

## 4. Sécurité et Gestion de la VRAM GPU

La mémoire vidéo (VRAM) est une ressource très limitée sur GPU. Nous avons implémenté des règles de gestion strictes pour stabiliser la VRAM tout au long de l'entraînement :

1.  **Limitation de la taille de lot (Batch Size = 64)** : Cette valeur a été choisie pour saturer modérément les cœurs CUDA sans dépasser la mémoire physique disponible.
2.  **Nettoyage Actif du Cache CUDA** :
    PyTorch utilise un allocateur de mémoire interne par blocs (Caching Allocator) pour éviter le coût temporel d'allocations répétées auprès du système d'exploitation du GPU. Cependant, cet allocateur a tendance à conserver des blocs de mémoire virtuellement "libres" mais non restitués au GPU, ce qui peut provoquer des erreurs d'OOM lors d'évaluations lourdes.
    Pour corriger cela, nous exécutons une routine de nettoyage explicite à la fin de **chaque époque d'entraînement** et après chaque boucle de validation :
    ```python
    # Vider la mémoire non référencée en RAM
    gc.collect()
    # Forcer l'allocateur PyTorch à restituer toute la mémoire libre au GPU
    torch.cuda.empty_cache()
    ```
    Cette routine garantit que la consommation de VRAM reste stable et ne présente aucune dérive mémoire (memory leak) au fil des heures d'entraînement.

---

## 5. Analyse de l'Empreinte Mémoire du Pipeline

| Composant | Taille sur Disque | Utilisation RAM active en Entraînement | Empreinte Mémoire Globale |
| :--- | :---: | :---: | :---: |
| **VoxCeleb Speech** (Parole) | ~33 Go | **~100 Mo** (Uniquement le batch en cours de chargement) | 🟢 Négligeable |
| **MUSAN Noise** (Bruit) | ~11 Go | **~20 Mo** (Uniquement le bruit du batch actif) | 🟢 Négligeable |
| **Poids des Modèles** | N/A | **~250 Mo** (CPU) / **~1,5 Go** (VRAM active avec gradients) | 🟢 Très faible |
| **Total du Pipeline** | **~44 Go** | **~2,5 Go à 3 Go de RAM système** | 🟢 **100% Sûr pour les limites Kaggle** |

---

## 6. Passage à l'Échelle : Entraînement sur l'ensemble des Locuteurs

Grâce à cette architecture à chargement dynamique, **la consommation de mémoire RAM et VRAM est totalement indépendante du nombre total de locuteurs présents dans la base de données**. Elle ne dépend que de la taille du lot (Batch Size) et de la taille du modèle.

Pour passer d'un entraînement de test à 50 locuteurs à un entraînement final à **1 111 locuteurs**, il a suffi de modifier la variable de contrôle à la racine du code :
```python
# Sélectionner l'ensemble complet des locuteurs VoxCeleb
SELECT_SUBSET_SPEAKERS = None
```
Ce changement a permis au script de scanner et d'intégrer l'ensemble des dossiers de locuteurs disponibles sur Kaggle, tout en fonctionnant de manière parfaitement fluide et stable dans les mêmes limites de mémoire ($3$ Go de RAM et $1{,}5$ Go de VRAM).
