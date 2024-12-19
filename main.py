from fonctions import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    login_wandb,
    download_csv_from_drive,
    load_and_prepare_data,
    preprocess_data,
    configure_training,
    train_and_save_model,
)

# Fonction principale
def main():
    # Lien Google Drive du fichier CSV
    google_drive_link = "https://drive.google.com/file/d/15lmWc_nmDQCasDuW-OK8mKxuczJ0DZXU/view?usp=sharing"  # Remplacez par le lien partageable de votre fichier CSV
    csv_path = "rakitra.csv"  # Chemin local où le fichier CSV sera sauvegardé
    output_dir = "./output"  # Dossier de sortie pour les modèles et logs

    # Télécharger le fichier CSV depuis Google Drive
    print("Téléchargement du fichier CSV depuis Google Drive...")
    downloaded_csv_path = download_csv_from_drive(google_drive_link, csv_path)
    print(f"Fichier CSV téléchargé avec succès : {downloaded_csv_path}")

    # Connexion à W&B
    print("Connexion à Weights & Biases...")
    login_wandb()

    # Charger le tokenizer et le modèle
    print("Chargement du tokenizer et du modèle GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Définir le token de padding
    tokenizer.pad_token = tokenizer.eos_token  # Utiliser le token EOS comme token de padding

    # Charger et préparer les données
    print("Chargement et préparation des données...")
    train_dataset, eval_dataset = load_and_prepare_data(downloaded_csv_path)

    # Prétraiter les données
    print("Prétraitement des données...")
    train_dataset, eval_dataset = preprocess_data(train_dataset, eval_dataset, tokenizer)

    # Configurer l'entraînement
    print("Configuration de l'entraînement...")
    trainer = configure_training(model, train_dataset, eval_dataset, tokenizer, output_dir)

    # Lancer l'entraînement et sauvegarder le modèle
    print("Lancement de l'entraînement...")
    train_and_save_model(trainer, output_dir)

    print("Pipeline terminé avec succès.")

# Appeler la fonction principale
if __name__ == "__main__":
    main()