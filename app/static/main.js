/**
 * Main JavaScript for ML Retail Prediction Interface
 */

document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('predictionForm');
    const resultsContainer = document.getElementById('resultsContainer');
    const initialMessage = document.getElementById('initialMessage');
    const loadingSpinner = document.getElementById('loadingSpinner');

    // Événement de soumission du formulaire
    form.addEventListener('submit', async function (e) {
        e.preventDefault();

        // Collecter les données du formulaire
        const formData = new FormData(form);
        const data = Object.fromEntries(formData);

        // Validation simple - seulement vérifier que les 4 champs principaux sont remplis et valides
        const requiredFields = ['Recency', 'Frequency', 'MonetaryTotal', 'Age'];

        for (const field of requiredFields) {
            const value = data[field];
            if (value === '' || value === null || value === undefined) {
                showError(`Veuillez remplir le champ: ${field}`);
                return;
            }
            if (isNaN(parseFloat(value)) || parseFloat(value) < 0) {
                showError(`Le champ "${field}" doit être un nombre positif`);
                return;
            }
        }

        // Montrer le spinner
        loadingSpinner.style.display = 'block';
        resultsContainer.style.display = 'none';
        initialMessage.style.display = 'none';

        try {
            // Envoyer la prédiction
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (result.success) {
                // Afficher les résultats
                displayResults(result.predictions, result.interpretations);
                resultsContainer.style.display = 'block';
            } else {
                // Afficher l'erreur
                showError(result.error || 'Erreur lors de la prédiction');
            }
        } catch (error) {
            showError('Erreur réseau: ' + error.message);
        } finally {
            loadingSpinner.style.display = 'none';
        }
    });

    // Fonction d'affichage des résultats
    function displayResults(predictions, interpretations) {
        // Churn
        const churnPercentage = predictions.churn_probability;
        let churnStatus = '';
        let churnColor = '';

        if (churnPercentage > 40) {
            churnStatus = 'RISQUE ÉLEVÉ';
            churnColor = '#dc3545'; // Rouge
        } else if (churnPercentage > 15) {
            churnStatus = 'À SURVEILLER';
            churnColor = '#fd7e14'; // Orange
        } else {
            churnStatus = 'CLIENT FIDÈLE';
            churnColor = '#28a745'; // Vert
        }

        document.getElementById('churnStatus').textContent = churnStatus;
        document.getElementById('churnStatus').style.color = churnColor;
        document.getElementById('churnPercentage').textContent = churnPercentage.toFixed(1) + '%';
        document.getElementById('churnProgressBar').style.width = churnPercentage.toFixed(1) + '%';
        document.getElementById('churnProgressBar').style.backgroundColor = churnColor;
        document.getElementById('churnHeader').style.borderBottom = `5px solid ${churnColor}`;

        // Segmentation
        const profile = interpretations.segment_profile;
        document.getElementById('segmentName').textContent = profile.name;
        document.getElementById('segmentEmoji').textContent = profile.emoji;

        // Dépense
        const spend = predictions.predicted_spend;
        document.getElementById('predictedSpend').textContent = '£' + spend.toFixed(2);
        document.getElementById('spendCategory').textContent = interpretations.spend_category;
    }


    // Fonction pour afficher les erreurs
    function showError(message) {
        alert('❌ Erreur: ' + message);
        resultsContainer.style.display = 'none';
        initialMessage.style.display = 'block';
    }
});

// Fonction pour réinitialiser le formulaire
function resetForm() {
    document.getElementById('predictionForm').reset();
    document.getElementById('resultsContainer').style.display = 'none';
    document.getElementById('initialMessage').style.display = 'block';
}
