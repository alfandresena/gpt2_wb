import os
import wandb
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset
import gdown

# Fonction pour initialiser Weights & Biases
def login_wandb():
    wandb.login(key="5b34fa313ec0e49c35fff4a66fc53cfbf0aac64c")  # Connexion à Weights & Biases

# Fonction pour télécharger un fichier CSV depuis Google Drive
def download_csv_from_drive(google_drive_link, output_path):
    """
    Télécharge un fichier CSV depuis Google Drive à l'aide d'un lien partageable.
    
    :param google_drive_link: Lien partageable Google Drive (doit contenir le paramètre `id=...` ou `file/d/...`).
    :param output_path: Chemin local où le fichier téléchargé sera sauvegardé.
    :return: Le chemin du fichier téléchargé.
    """
    # Extraire l'ID du fichier à partir du lien
    if "id=" in google_drive_link:
        file_id = google_drive_link.split("id=")[1]
    elif "file/d/" in google_drive_link:
        file_id = google_drive_link.split("file/d/")[1].split("/")[0]
    else:
        raise ValueError("Lien Google Drive non valide.")
    
    # Construire l'URL de téléchargement direct
    download_url = f"https://drive.google.com/uc?id={file_id}"
    
    # Télécharger le fichier
    gdown.download(download_url, output_path, quiet=False)
    
    return output_path

# Fonction pour charger et préparer les données
def load_and_prepare_data(data_path):
    dataset = load_dataset("csv", data_files={"train": data_path})
    train_test_split = dataset["train"].train_test_split(test_size=0.1)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]
    return train_dataset, eval_dataset

# Fonction de tokenisation des données
def preprocess_data(train_dataset, eval_dataset, tokenizer):
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )
        tokenized["labels"] = tokenized["input_ids"].copy()  # Ajouter les labels
        return tokenized

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    return train_dataset, eval_dataset

# Fonction pour configurer l'entraînement
def configure_training(model, train_dataset, eval_dataset, tokenizer, output_dir):
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",  # Évaluation périodique
        eval_steps=500,  # Fréquence des évaluations
        save_steps=500,
        per_device_train_batch_size=8,
        num_train_epochs=1,
        save_total_limit=2,  # Limiter les checkpoints sauvegardés
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,  # Fréquence des logs
        report_to=["wandb"],  # Activer Weights & Biases
        run_name="gpt2-training-run"  # Nom de l'exécution dans W&B
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,  # Pour gérer le padding
    )
    
    return trainer

# Fonction pour lancer l'entraînement et sauvegarder le modèle
def train_and_save_model(trainer, output_dir):
    trainer.train()  # Lancer l'entraînement
    trainer.save_model()  # Sauvegarder le modèle
    trainer.tokenizer.save_pretrained(output_dir)  # Sauvegarder le tokenizer
    print("Entraînement terminé et modèle sauvegardé dans:", output_dir)

# Exemple d'utilisation
if __name__ == "__main__":
    google_drive_link = "LIEN_PARTAGEABLE_GOOGLE_DRIVE"
    csv_path = "data.csv"  # Chemin local pour sauvegarder le fichier téléchargé

    # Télécharger le CSV depuis Google Drive
    downloaded_csv_path = download_csv_from_drive(google_drive_link, csv_path)
    print(f"Fichier CSV téléchargé: {downloaded_csv_path}")

    # Charger et préparer les données
    train_dataset, eval_dataset = load_and_prepare_data(downloaded_csv_path)