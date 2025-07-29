# Adversarial Example Generation for Audio-based Models Using Whisper
Autor: Lukáš Hofman  
Jedná se o kód spjatý s bakalářskou prací "Nepřátelské vzory pro modely rozpoznávání řeči".

Tato práce využívá whisper_attack a robust_speech, dostupné pod Apache License 2.0 (http://www.apache.org/licenses/) a model Whisper od OpenAI dostupný pod MIT License.

Z projektu whisper_attack je vybrána pouze podmnožina souborů a některé z nich obsahují malé úpravy. 
Všechny úpravy v .py souborech u sebe mají komentář "# ADDED" nebo "# CHANGED". .yaml soubory a bashovské skripty žádnou poznámku kvůli změnám nemají, ale mohly být upraveny v zájmu testování. Některé .yaml soubory jsou v attack_configs/whisper/ nové, při jejich tvorbě jsem se držel formátu z whisper_attack.   

## Overview

Tento projekt se zaměřuje na tvorbu nepřátelských vzorů pro model rozpoznání řeči Whisper od společnosti OpenAI. Nepřátelské vzory jsou vstupy specificky vytvořené tak, aby zmátly modely umělé inteligence. Tento repozitář demonstruje jak takové nepřátelské vzory vytvořit a umožňuje je aplikovat na model Whisper.

Projekt byl testován na WSL 2 Ubuntu 24.04.1 LTS.
Aplikace byla testována na verzi Pythonu 3.10 kvůli kompatibilitě s modely Whisper a knihovnami, v praxi by mělo být možné rozběhnout aplikaci už od verze Pythonu 3.8.

Pro replikaci výsledků proveďte následující:

### Virtuální prostředí
Vytvořte virtuální prostředí:
```bash
python3.10 -m venv Python310
source Python310/bin/activate
pip install -r requirements.txt
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```
Verze vaší GPU se může lišit, nalezněte nejpodobnější verzi torch k verzi 2.1.0 podporující vaši GPU.  

robust_speech knihovna oficiálně požaduje torch>=1.8.0,<=1.11, ale funguje i na vyšších verzích. Proto se nezalekněte následujícího erroru:  

"ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
robust-speech 0.3.1 requires torch<=1.11,>=1.8.0, but you have torch 2.1.0+cu118 which is incompatible."  


Nakonec je potřeba stáhnout Tkinter:
```bash
sudo apt-get install python3-tk
```

### Stáhnutí datasetů
```bash
cd path_to_git_directory/data/
wget https://www.openslr.org/resources/12/test-clean.tar.gz
tar -xzf test-clean.tar.gz
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
tar -xzf train-clean-100.tar.gz
cd ..
```

Poté spusťte prepare_datasets.py se správnou cestou
```bash
python prepare_datasets.py --libri_speech_path="path_to_git_directory/data/LibriSpeech"
```

Pokud chcete podmnožinu celého datasetu, využijte create_subset.py.
Příklad použití:
```bash
python create_subset.py --csv_path="path_to_git_directory/data/LibriSpeech/csv/test-clean.csv" --num_samples=100
```

#### Trénování univerzálních útoků
GUI ani CLI nástroje neposkytují podporu pro trénování univerzálních perturbací – slouží pouze k jejich aplikaci. Pokud však potřebujete vytvořit vlastní univerzální útok (např. s jinou hodnotou epsilon nebo pro s jiným zvukem na pozadí), je třeba spustit příslušný trénovací skript ručně pomocí bashového skriptu. K tvorbě univerzální perturbace využijte soubory *whisper_attack/fit_attacker.py* a *attack_configs/whisper/univ_pgd_fit.yaml*. (respektive *attack_configs/whisper/univ_noise_fit.yaml*) Parametry lze upravit buď přímo v YAML souborech, nebo je specifikovat jako argumenty v Bash skriptu při spuštění trénování.

Příklad bash skriptu trénujícího univerzálni PGD útoku:
```bash
#! /bin/bash
RSROOT=PLACEHOLDER
export PYTHONPATH="${PYTHONPATH}:${RSROOT}"

EPOCHS=100
NBITER=200
SEED=55
EVERY=10
BATCH=64
EPS=0.02
EPS_ITEMS=0.1
RELEPS=0.05

LOAD=False
DATA=test-clean-100
TRAIN_DATA=train-clean-100-128

python whisper_attack/fit_attacker.py attack_configs/whisper/univ_pgd_fit.yaml --epochs=$EPOCHS --batch_size=$BATCH --data_csv_name=$DATA --data_csv_name_train=$TRAIN_DATA --model_label=tiny --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --eps=$EPS --rel_eps_iter=$RELEPS --success_every=$EVERY
python whisper_attack/fit_attacker.py attack_configs/whisper/univ_pgd_fit.yaml --epochs=$EPOCHS --batch_size=$BATCH --data_csv_name=$DATA --data_csv_name_train=$TRAIN_DATA --model_label=base --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --eps=$EPS --rel_eps_iter=$RELEPS --success_every=$EVERY
python whisper_attack/fit_attacker.py attack_configs/whisper/univ_pgd_fit.yaml --epochs=$EPOCHS --batch_size=$BATCH --data_csv_name=$DATA --data_csv_name_train=$TRAIN_DATA --model_label=small --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --eps=$EPS --rel_eps_iter=$RELEPS --success_every=$EVERY
python whisper_attack/fit_attacker.py attack_configs/whisper/univ_pgd_fit.yaml --epochs=$EPOCHS --batch_size=$BATCH --data_csv_name=$DATA --data_csv_name_train=$TRAIN_DATA --model_label=medium --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --eps=$EPS --rel_eps_iter=$RELEPS --success_every=$EVERY
```

Natrénovaná perturbace je uložena do .ckpt souboru ve složce definované v attack_configs/whisper/univ_pgd_fit.yaml (nebo attack_configs/whisper/univ_noise_fit.yaml)


## Funkce
- **Konzolové prostředí pro datasety**
- **Grafické uživatelské prostředí**

## Konzolové prostředí pro tvorbu a testování útoků
Dataset, který používám, je test-clean a train-clean-100 od LibriSpeech, ale je možné použít jakýkoliv jiný. Pokud ovšem chcete jiný použít, je potřeba upravit .yaml soubory útoků (nebo přepsat parametry v argumentu overrides).
Pro změnu v .yaml souboru je třeba upravit tuto část:
```
# Data files
data_folder: !ref <root>/data/LibriSpeech # e.g, /localscratch/LibriSpeech
csv_folder: !ref <data_folder>/csv # e.g, /localscratch/LibriSpeech
# If RIRS_NOISES dir exists in /localscratch/xxx_corpus/RIRS_NOISES
# then data_folder_rirs should be /localscratch/xxx_corpus
# otherwise the dataset will automatically be downloaded
test_splits: ["test-clean"]
skip_prep: True
ckpt_interval_minutes: 15 # save checkpoint every N min
data_csv_name: test-clean
```

### 1. Spuštění konzolové aplikace:
```bash
python console_interface.py
```
Po spuštění aplikace by se vám mělo zobrazit
```
Welcome to Whisper Attack Console. Type 'help' for commands.
(attack) > 
```
### 2. Použití
#### Příkazy

| Příkaz                 | Popis                          | Příklad                     |
|------------------------|--------------------------------|-----------------------------|
| `model`                | Nastavení velikosti modelu     | `model base`                |
| `dataset`              | Volba datasetu                 | `dataset test-clean-100`    |
| `seed`                 | Nastavení náhodného seedu      | `seed 42`                   |
| `load_audio`           | Načítání audio (True/False)    | `load_audio True`           |
| `do_skip_prep_dataset` | Příprava datasetu (True/False) | `skip_prep_dataset n`       |
| `attack`               | Spuštění útoku                 | `attack snr_pgd`            |
| `exit`                 | Ukončení programu              | `exit`                      |

#### Dostupné útoky

| Název útoku       | Popis                                            |
|-------------------|--------------------------------------------------|
| `snr_pgd`         | Útok optimalizující SNR                          |
| `pgd`             | Základní PGD útok                                |
| `cw`              | Carlini-Wagner útok                              |
| `cw_modified`     | Modifikovaný Carlini-Wagner útok                 |
| `genetic`         | Genetický algoritmus                             |
| `rand`            | Náhodný šum                                      |
| `universal`       | Univerzální PGD útok                             |
| `universal_noise` | Univerzální PGD útok s přidaným zvukem na pozadí |


#### Výchozí hodnoty

| Parametr      | Vhodnota        |
|---------------|-----------------|
| Model         | `base`          |
| Dataset       | `test-clean-100`|
| Seed          | `250`           |

### 3. Příklad užití
(attack) > model small
Model set to: small
(attack) > attack snr_pgd
Number of iterations (default 200): 
Using default value for Number of iterations: 200
Batch size (default 1): 2
Using value for Batch size: 2
SNR (default 35): 
Using default value for SNR: 35 

#### 4. Ukončení konzolové aplikace:
Pro ukončení napište do konzole exit.
```
(attack) > exit
Exiting...
```

Pokud chcete spustit více útoků s rozdílnými parametry jedním rozkazem, můžete na to vytvořit bash skript podobně jako na trénování univerzálních útoků.
Například:
```bash
#! /bin/bash
# genetic.sh
RSROOT=!PLACEHOLDER
export PYTHONPATH="${PYTHONPATH}:${RSROOT}"

SEED=200
NBITER=200
LOAD=False
DATA=test-clean-20
NAME=genetic

python whisper_attack/run_attack.py attack_configs/whisper/genetic.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=tiny --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME
# python whisper_attack/run_attack.py attack_configs/whisper/genetic.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=base --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME
# python whisper_attack/run_attack.py attack_configs/whisper/genetic.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=small --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME
# python whisper_attack/run_attack.py attack_configs/whisper/genetic.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=medium --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME

# python whisper_attack/run_attack.py attack_configs/whisper/geneticTargeted.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=tiny --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME
# python whisper_attack/run_attack.py attack_configs/whisper/geneticTargeted.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=base --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME
# python whisper_attack/run_attack.py attack_configs/whisper/geneticTargeted.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=small --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME
# python whisper_attack/run_attack.py attack_configs/whisper/geneticTargeted.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=medium --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME
```
a poté ho jednoduše spustit:
```bash
bash genetic.sh
```

### Grafické uživatelské prostředí

#### Přehled

Tento GUI nástroj poskytuje grafické rozhraní pro vytváření adversariálních příkladů na jednom zvukovém souboru pomocí modelů typu Whisper (tiny, base, small, medium, large).

Podporované typy útoků:

* PGD
* SNR PGD
* Carlini-Wagner (CW)
* Modifikovaný CW
* Genetický útok
* Gaussovský šum
* Univerzální PGD útok
* Univerzální PGD útok s přidaným zvukem na pozadí

#### Hlavní funkcionalita

* Načtení .wav souboru ze souborového systému
* Přepis originálního i adversariálního zvuku modelem Whisper
* Přehrávání originálního i upraveného zvuku
* Uložení výsledného zvuku
* Podpora pro parametrizaci útoku (epsilon, iterace, SNR, atd.)
* Volba velikosti modelu: tiny, base, small, medium, large
* Možnost volby cílového textu (targeted attack)

#### Použití

Pokud používáte WSL 2, WSL automaticky nepodporuje audio zařízení, takže je musítě prvně zprovoznit. 
Odkaz na zprovoznění: https://research.wmz.ninja/articles/2017/11/setting-up-wsl-with-graphics-and-audio.html

1. Spusťte aplikaci:

```bash
python GUI.py
```
2. Načtěte zvukový záznam:
Pomocí tlačítka "Load Audio" načtěte zvukový soubor ve formátu `.wav`.
Poté můžete zvuk přehrát a zkontrolovat, zda jste vybrali požadovaný zvukový soubor, kliknutím na tlačítko **Play original audio**.
3. Vyberte model a typ útoku
Na levém menu vyberte, zda chcete vytvořit cílený útok, a poté, který typ útoku chcete aplikovat.
4. Nastavte parametry útoku
Pozor, parametry útoků se liší útok od útoku.
5. Klikněte na **Generate example**
Tím se spustí skutečný útok a vy budete muset počkat, až se dokončí.
Po jeho dokončení se vypíše původní přepis, přepis nepřátelského vzoru a budete mít možnost přehrát rozšířený zvuk pro kontrolu kvality zvuku.
Pokud chcete, můžete nepřátelský útok uložit klepnutím na tlačítko **Save example**.
6. Ukončete aplikaci
Pokud chcete aplikaci ukončit, klikněte na tlačítko **Exit**.

#### Struktura GUI

* `parameters_frame`: Rámec pro parametry útoku.
* `original_frame`: Rámec pro původní nahrávku.
* `generate_button_frame`: Rámec pro tlačítko spuštění útoku.
* `generated_frame`: Rámec pr transkripce nahrávek a tlačítka pro přehrání / uložení.
* `targeted_frame`: Rámec pro cílený útok a jeho cíl.
* `exit_frame`: Rámec pro tlačítko pro ukončení.

#### Detaily

* Výběr typu útoku volá `on_attack_type_change`, který dynamicky zobrazí relevantní parametry útoku
* Volba modelu volá `load_model.load_whisper_model()`, který načte odpovídající velikost modelu Whisper. 
* Adversariální zvuk je vytvořen funkcí `generate_example`
* Zvuk je hrán přes `pygame.mixer`
* Nepřátelský vzor je generován pomocí `create_adversarial_attack`. 

#### Omezení

* Funguje pouze na `.wav` soubory
* Neumožňuje batch processing
* Nenabízí vizualizaci spectrogramů nebo výsledků
