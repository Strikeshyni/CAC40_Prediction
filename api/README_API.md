# CAC40 Stock Prediction API 🚀

API REST complète pour l'entraînement de modèles ML de prédiction de prix d'actions et la simulation de stratégies de trading avec 5 stratégies différentes.

## 🆕 Nouveautés v2.0

### 5 Stratégies de Trading
- **Simple** : Achat/vente basique selon les prédictions
- **Threshold** : Trade uniquement si la différence dépasse un seuil (€)
- **Percentage** : Trade basé sur le pourcentage de changement (%)
- **Conservative** : Attend un profit cible avant de vendre
- **Aggressive** : Trading agressif avec stop-loss automatique

### Suivi Détaillé des Transactions
- ✅ Historique complet de toutes les transactions
- ✅ Détails : prix, quantité, raison d'achat/vente, profit/perte
- ✅ Métriques en temps réel : balance, stocks, portfolio value
- ✅ WebSocket pour suivi en direct des simulations

### Nouveaux Endpoints
- `GET /api/simulate/{sim_id}/status` - Statut de simulation
- `GET /api/simulate/{sim_id}/transactions` - Historique des transactions
- `GET /api/simulate/{sim_id}/results` - Résultats complets
- `GET /api/simulate/jobs` - Liste toutes les simulations
- `WS /ws/simulation/{sim_id}` - WebSocket pour progression en temps réel

## 📦 Installation

```bash
# Installer les dépendances de l'API
pip install -r api/requirements_api.txt

# Les dépendances principales du projet doivent déjà être installées
pip install -r requirements.txt
```

## 🚀 Démarrage Rapide

```bash
# Démarrer le serveur API
cd /home/abel/personnal_projects/CAC40_stock_prediction
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8002
```

L'API sera accessible sur `http://localhost:8002`

**Documentation interactive** : `http://localhost:8002/docs`

### Test Rapide
```bash
# Tester avec le script d'exemple
python api/api_example_client.py
```

## 📚 Documentation Complète

- **[QUICKSTART.md](QUICKSTART.md)** - Guide de démarrage rapide avec exemples
- **[STRATEGIES_GUIDE.md](STRATEGIES_GUIDE.md)** - Guide détaillé des 5 stratégies de trading
- **[api_example_client.py](api_example_client.py)** - Script Python de test complet

## 🔌 Endpoints

### 1. Health Check
```bash
GET /health
```

### 2. Entraîner un modèle
```bash
POST /api/train
Content-Type: application/json

{
  "stock_name": "ENGI.PA",
  "from_date": "2015-01-01",
  "to_date": "2025-01-01",
  "train_size_percent": 0.8,
  "val_size_percent": 0.2,
  "time_step": 300,
  "global_tuning": true
}
```

**Réponse :**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "Training job started successfully",
  "config": {...}
}
```

### 3. Vérifier le statut de l'entraînement
```bash
GET /api/train/{job_id}/status
```

**Réponse :**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "progress": 0.65,
  "current_step": "Training model with hyperparameter tuning",
  "start_time": "2025-05-20T10:00:00",
  "end_time": null,
  "error": null,
  "model_path": null
}
```

**Statuts possibles :**
- `pending` : En attente de démarrage
- `running` : En cours d'exécution
- `completed` : Terminé avec succès
- `failed` : Échec

### 4. Suivre l'entraînement en temps réel (WebSocket)
```javascript
// Exemple JavaScript
const ws = new WebSocket('ws://localhost:8002/ws/training/{job_id}');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(`Progress: ${data.progress * 100}%`);
  console.log(`Step: ${data.current_step}`);
};
```

### 5. Faire des prédictions
```bash
POST /api/predict
Content-Type: application/json

{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "n_days": 5
}
```

**Réponse :**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "stock_name": "ENGI.PA",
  "predictions": [
    {"day": 1, "predicted_price": 12.45},
    {"day": 2, "predicted_price": 12.52},
    {"day": 3, "predicted_price": 12.48},
    {"day": 4, "predicted_price": 12.60},
    {"day": 5, "predicted_price": 12.55}
  ],
  "last_actual_price": 12.40,
  "last_actual_date": "2025-01-01"
}
```

### 6. Lancer une simulation historique (NOUVEAU)
```bash
POST /api/simulate
Content-Type: application/json

{
  "stock_name": "ENGI.PA",
  "from_date": "2024-01-01",
  "to_date": "2024-12-31",
  "initial_balance": 100.0,
  "time_step": 300,
  "nb_years_data": 10,
  
  // Stratégie et paramètres (NOUVEAU)
  "strategy": "conservative",           // simple, threshold, percentage, conservative, aggressive
  "buy_threshold": 2.0,                 // Seuil d'achat (dépend de la stratégie)
  "sell_threshold": 1.5,                // Seuil de vente
  "min_profit_percentage": 5.0,         // Profit minimum avant vente (conservative)
  "max_loss_percentage": 3.0            // Stop-loss (aggressive)
}
```

**Réponse immédiate (202 Accepted) :**
```json
{
  "sim_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "pending",
  "stock_name": "ENGI.PA",
  "simulation_period": {
    "from": "2024-01-01",
    "to": "2024-12-31"
  },
  "initial_balance": 100.0,
  "strategy_used": "conservative"
}
```

### 7. Vérifier le statut de la simulation (NOUVEAU)
```bash
GET /api/simulate/{sim_id}/status
```

**Réponse :**
```json
{
  "sim_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "running",
  "progress": 0.65,
  "current_date": "2024-08-15",
  "days_processed": 195,
  "total_days": 300,
  "current_balance": 115.30,
  "current_stocks": 8.45,
  "total_transactions": 23,
  "start_time": "2025-05-20T10:00:00",
  "end_time": null,
  "error": null
}
```

### 8. Récupérer les transactions (NOUVEAU)
```bash
GET /api/simulate/{sim_id}/transactions
```

**Réponse :**
```json
{
  "sim_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "total_transactions": 23,
  "transactions": [
    {
      "transaction_id": 1,
      "date": "2024-01-05",
      "transaction_type": "buy",
      "stock_price": 12.30,
      "quantity": 8.13,
      "total_value": 100.00,
      "balance_after": 0.0,
      "stocks_owned_after": 8.13,
      "reason": "Conservative buy: +2.5% predicted (threshold: 2.0%)",
      "predicted_price": 12.61,
      "predicted_change_pct": 2.52
    },
    {
      "transaction_id": 2,
      "date": "2024-01-12",
      "transaction_type": "sell",
      "stock_price": 12.92,
      "quantity": 8.13,
      "total_value": 105.04,
      "balance_after": 105.04,
      "stocks_owned_after": 0.0,
      "reason": "Profit target reached: 5.04% (target: 5.0%)",
      "predicted_price": 12.85,
      "predicted_change_pct": -0.54
    }
  ],
  "summary": {
    "total_transactions": 23,
    "buy_transactions": 12,
    "sell_transactions": 11,
    "total_invested": 1200.00,
    "total_returned": 1305.50,
    "net_trading_result": 105.50
  }
}
```

### 9. Récupérer les résultats complets (NOUVEAU)
```bash
GET /api/simulate/{sim_id}/results
```

**Réponse (uniquement quand status = "completed") :**
```json
{
  "sim_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "completed",
  "stock_name": "ENGI.PA",
  "simulation_period": {
    "from": "2024-01-01",
    "to": "2024-12-31"
  },
  "initial_balance": 100.0,
  "final_balance": 125.50,
  "benefit": 25.50,
  "benefit_percentage": 25.5,
  "strategy_used": "conservative",
  "daily_results": [...],
  "transactions": [...],
  "summary": {
    "total_trades": 23,
    "buy_trades": 12,
    "sell_trades": 11,
    "winning_trades": 8,
    "losing_trades": 3,
    "win_rate": 72.73,
    "total_days": 300,
    "days_with_errors": 0
  }
}
```

### 10. Lister toutes les simulations (NOUVEAU)
```bash
GET /api/simulate/jobs
```

### 11. Supprimer une simulation (NOUVEAU)
```bash
DELETE /api/simulate/{sim_id}
```

### 12. Lister tous les entraînements
```bash
GET /api/train/jobs
```

### 13. Supprimer un entraînement
```bash
DELETE /api/train/{job_id}
```

### 14. WebSocket pour simulations (NOUVEAU)
```javascript
const ws = new WebSocket('ws://localhost:8002/ws/simulation/{sim_id}');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(`Progress: ${(data.progress * 100).toFixed(1)}%`);
  console.log(`Current date: ${data.current_date}`);
  console.log(`Balance: ${data.current_balance.toFixed(2)}€`);
  console.log(`Stocks: ${data.current_stocks.toFixed(4)}`);
  console.log(`Total transactions: ${data.total_transactions}`);
  
  if (data.status === 'completed') {
    console.log('Simulation terminée!');
    ws.close();
  }
};
```

## 📊 Exemples d'Utilisation

### Exemple 1 : Simulation Simple
```python
import requests
import time

# URL de l'API
API_URL = "http://localhost:8002"

# 1. Entraîner un modèle
config = {
    "stock_name": "ENGI.PA",
    "from_date": "2015-01-01",
    "to_date": "2024-12-31",
    "train_size_percent": 0.8,
    "val_size_percent": 0.2,
    "time_step": 300,
    "global_tuning": True
}

response = requests.post(f"{API_URL}/api/train", json=config)
job_id = response.json()["job_id"]
print(f"Job ID: {job_id}")

# 2. Vérifier le statut
while True:
    status_response = requests.get(f"{API_URL}/api/train/{job_id}/status")
    status = status_response.json()
    
    print(f"Status: {status['status']} - Progress: {status['progress']*100:.1f}%")
    print(f"Step: {status['current_step']}")
    
    if status['status'] in ['completed', 'failed']:
        break
    
    time.sleep(5)

# 3. Faire des prédictions
if status['status'] == 'completed':
    pred_request = {
        "job_id": job_id,
        "n_days": 5
    }
    
    pred_response = requests.post(f"{API_URL}/api/predict", json=pred_request)
    predictions = pred_response.json()
    
    print("\nPrédictions:")
    for pred in predictions['predictions']:
        print(f"Jour {pred['day']}: {pred['predicted_price']:.2f}€")
```

### Exemple 2 : Simulation avec Stratégie (NOUVEAU)
```python
import requests
import time

API_URL = "http://localhost:8002"

# Configuration avec stratégie conservative
config = {
    "stock_name": "ENGI.PA",
    "from_date": "2024-01-01",
    "to_date": "2024-11-20",
    "initial_balance": 100.0,
    "strategy": "conservative",
    "min_profit_percentage": 5.0,  # Vendre uniquement si profit >= 5%
    "buy_threshold": 2.0            # Acheter si hausse prédite >= 2%
}

# Lancer la simulation
response = requests.post(f"{API_URL}/api/simulate", json=config)
sim_id = response.json()["sim_id"]
print(f"Simulation ID: {sim_id}")

# Suivre la progression
while True:
    status = requests.get(f"{API_URL}/api/simulate/{sim_id}/status").json()
    print(f"Progress: {status['progress']*100:.1f}% - "
          f"Date: {status['current_date']} - "
          f"Balance: {status['current_balance']:.2f}€ - "
          f"Transactions: {status['total_transactions']}")
    
    if status['status'] in ['completed', 'failed']:
        break
    time.sleep(2)

# Récupérer les résultats
if status['status'] == 'completed':
    # Résultats complets
    results = requests.get(f"{API_URL}/api/simulate/{sim_id}/results").json()
    print(f"\n--- RÉSULTATS ---")
    print(f"Balance finale: {results['final_balance']:.2f}€")
    print(f"Profit/Perte: {results['benefit']:+.2f}€ ({results['benefit_percentage']:+.2f}%)")
    print(f"Stratégie: {results['strategy_used']}")
    print(f"Trades: {results['summary']['total_trades']}")
    print(f"Win rate: {results['summary']['win_rate']:.1f}%")
    
    # Transactions détaillées
    transactions = requests.get(f"{API_URL}/api/simulate/{sim_id}/transactions").json()
    print(f"\n--- DERNIÈRES TRANSACTIONS ---")
    for t in transactions['transactions'][-5:]:
        print(f"{t['date']} | {t['transaction_type'].upper():4s} | "
              f"{t['quantity']:.2f} @ {t['stock_price']:.2f}€")
        print(f"  → {t['reason']}")
```

### Exemple 3 : Comparer Plusieurs Stratégies (NOUVEAU)
```python
import requests
import time

API_URL = "http://localhost:8002"

strategies = [
    {"name": "Simple", "config": {"strategy": "simple"}},
    {"name": "Threshold 1€", "config": {"strategy": "threshold", "buy_threshold": 1.0, "sell_threshold": 1.0}},
    {"name": "Conservative 5%", "config": {"strategy": "conservative", "min_profit_percentage": 5.0}},
    {"name": "Aggressive 3%", "config": {"strategy": "aggressive", "max_loss_percentage": 3.0}},
]

results = []

for strat in strategies:
    config = {
        "stock_name": "ENGI.PA",
        "from_date": "2024-01-01",
        "to_date": "2024-11-20",
        "initial_balance": 100.0,
        **strat["config"]
    }
    
    response = requests.post(f"{API_URL}/api/simulate", json=config)
    sim_id = response.json()["sim_id"]
    
    # Attendre la fin
    while True:
        status = requests.get(f"{API_URL}/api/simulate/{sim_id}/status").json()
        if status["status"] in ["completed", "failed"]:
            break
        time.sleep(2)
    
    if status["status"] == "completed":
        result = requests.get(f"{API_URL}/api/simulate/{sim_id}/results").json()
        results.append({
            "strategy": strat["name"],
            "benefit_pct": result["benefit_percentage"],
            "trades": result["summary"]["total_trades"],
            "win_rate": result["summary"]["win_rate"]
        })

# Afficher comparaison
print(f"\n{'Stratégie':<20s} | {'Profit':<12s} | {'Trades':<8s} | {'Win Rate':<10s}")
print("-" * 60)
for r in sorted(results, key=lambda x: x['benefit_pct'], reverse=True):
    print(f"{r['strategy']:<20s} | {r['benefit_pct']:+6.2f}%      | {r['trades']:6d} | {r['win_rate']:7.1f}%")
```

### Exemple cURL
```bash
# Entraîner un modèle
curl -X POST "http://localhost:8000/api/train" \
  -H "Content-Type: application/json" \
  -d '{
    "stock_name": "ENGI.PA",
    "from_date": "2015-01-01",
    "to_date": "2024-12-31",
    "train_size_percent": 0.8,
    "val_size_percent": 0.2,
    "time_step": 300,
    "global_tuning": true
  }'

# Vérifier le statut
curl "http://localhost:8000/api/train/{job_id}/status"

# Faire des prédictions
curl -X POST "http://localhost:8002/api/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "your-job-id",
    "n_days": 5
  }'

# Simulation historique
curl -X POST "http://localhost:8002/api/simulate" \
  -H "Content-Type: application/json" \
  -d '{
    "stock_name": "ENGI.PA",
    "from_date": "2024-01-01",
    "to_date": "2024-12-31",
    "initial_balance": 100.0,
    "time_step": 300,
    "nb_years_data": 10
  }'
```

## 🛡️ Gestion des erreurs

L'API gère plusieurs types d'erreurs :

### Erreurs de validation (400)
```json
{
  "error": "Invalid input",
  "detail": "to_date must be after from_date"
}
```

### Job non trouvé (404)
```json
{
  "detail": "Job 550e8400-e29b-41d4-a716-446655440000 not found"
}
```

### Erreur de rate limiting Yahoo Finance (500)
```json
{
  "error": "Internal server error",
  "detail": "No data retrieved for ENGI.PA from 2015-01-01 to 2024-12-31. This might be due to rate limiting or invalid stock symbol."
}
```

### Job non terminé (400)
```json
{
  "detail": "Job 550e8400-e29b-41d4-a716-446655440000 is not completed yet. Current status: running"
}
```

## 🔧 Configuration avancée

### Paramètres de l'entraînement

- **stock_name** : Symbole du stock (ex: ENGI.PA pour Engie sur Euronext Paris)
- **from_date / to_date** : Période d'entraînement (format YYYY-MM-DD)
- **train_size_percent** : Pourcentage de données pour l'entraînement (0.1-0.95)
- **val_size_percent** : Pourcentage de données pour la validation (0.05-0.9)
- **time_step** : Nombre de jours utilisés pour prédire le jour suivant (10-1000)
- **global_tuning** : Active l'optimisation des hyperparamètres (plus lent mais meilleur)

### Paramètres de simulation

#### Paramètres de base
- **stock_name** : Symbole du stock (ex: ENGI.PA)
- **from_date / to_date** : Période de simulation (YYYY-MM-DD)
- **initial_balance** : Capital de départ (défaut: 100.0€)
- **time_step** : Taille de la séquence pour prédiction (défaut: 300)
- **nb_years_data** : Années de données historiques pour entraînement (défaut: 10)

#### Paramètres de stratégie (NOUVEAU)
- **strategy** : Type de stratégie (`simple`, `threshold`, `percentage`, `conservative`, `aggressive`)
- **buy_threshold** : Seuil d'achat (signification varie selon la stratégie)
- **sell_threshold** : Seuil de vente
- **min_profit_percentage** : Profit minimum avant vente (stratégie `conservative`)
- **max_loss_percentage** : Stop-loss automatique (stratégie `aggressive`)

#### Valeurs recommandées par stratégie

| Stratégie | buy_threshold | sell_threshold | min_profit_percentage | max_loss_percentage |
|-----------|---------------|----------------|----------------------|---------------------|
| **simple** | - | - | - | - |
| **threshold** | 0.5€ - 2.0€ | 0.5€ - 2.0€ | - | - |
| **percentage** | 1.0% - 3.0% | 1.0% - 3.0% | - | - |
| **conservative** | 2.0% - 5.0% | - | 5.0% - 10.0% | - |
| **aggressive** | 0.1% - 0.5% | - | - | 3.0% - 5.0% |

## Structure des fichiers

```
api/
├── __init__.py
├── main.py              # Application FastAPI principale
├── models.py            # Modèles Pydantic pour validation
├── services.py          # Logique métier (entraînement, prédictions, simulations)
├── utils.py             # Fonctions utilitaires
├── requirements_api.txt # Dépendances API
└── README_API.md        # Cette documentation
```

## ✨ Fonctionnalités

### Entraînement de Modèles
✅ Configuration flexible des dates et paramètres  
✅ Optimisation d'hyperparamètres (optionnel)  
✅ Suivi en temps réel via WebSocket  
✅ Modèles persistants et réutilisables  

### Prédictions
✅ Prédictions multi-jours (1-30 jours)  
✅ Basées sur des modèles entraînés personnalisés  

### Simulations de Trading (NOUVEAU)
✅ **5 stratégies de trading** : Simple, Threshold, Percentage, Conservative, Aggressive  
✅ **Historique complet des transactions** : Prix, quantité, raison, profit/perte  
✅ **Métriques détaillées** : Win rate, nombre de trades, profit total  
✅ **Suivi en temps réel** : WebSocket pour progression live  
✅ **Paramètres configurables** : Seuils, stop-loss, profit target  

### Infrastructure
✅ **Gestion d'erreurs robuste** : Validation Pydantic, gestion rate limiting Yahoo  
✅ **Cache intelligent** : Données et modèles mis en cache  
✅ **Exécution asynchrone** : Simulations et entraînements en background  
✅ **API RESTful** : Documentation Swagger interactive  

## ⚠️ Notes Importantes

### Performance
- **Simulations longues** : Peuvent prendre du temps (un modèle entraîné par jour simulé)
- **Cache des modèles** : Les simulations répétées sur les mêmes périodes sont plus rapides
- **Mémoire** : Les modèles sont stockés en RAM (considérez un stockage externe en production)

### Limitations
- **Frais de transaction** : Non pris en compte dans les simulations
- **Slippage** : Non simulé (prix d'exécution = prix prédit)
- **Yahoo Finance** : Rate limiting possible (attendez 1-2h si erreur "Too Many Requests")
- **Backtesting** : Les performances passées ne garantissent pas les résultats futurs

### Bonnes Pratiques
1. **Testez d'abord sur des courtes périodes** (1 mois) pour valider la configuration
2. **Comparez plusieurs stratégies** sur la même période pour identifier la meilleure
3. **Analysez les transactions** pour comprendre le comportement de la stratégie
4. **Utilisez WebSocket** pour les simulations longues (>3 mois)
5. **Ajustez les paramètres** selon la volatilité du stock

## 🐛 Debugging

### Activer les logs détaillés
```bash
uvicorn api.main:app --reload --log-level debug --port 8002
```

### Vérifier le statut d'une tâche
```bash
# Entraînement
curl http://localhost:8002/api/train/{job_id}/status

# Simulation
curl http://localhost:8002/api/simulate/{sim_id}/status
```

### Problèmes courants

#### L'API ne démarre pas
```bash
# Vérifier que les dépendances sont installées
pip install -r api/requirements_api.txt

# Vérifier le port
lsof -i :8002  # Voir si le port est déjà utilisé
```

#### Simulation bloquée
```bash
# Vérifier les erreurs dans les logs du serveur
# Vérifier le statut détaillé
curl http://localhost:8002/api/simulate/{sim_id}/status | jq
```

#### Rate Limit Yahoo Finance
- **Solution** : Attendre 1-2 heures avant de relancer
- **Alternative** : Utiliser des périodes déjà mises en cache

## 📞 Support

### Documentation
1. **Documentation interactive** : http://localhost:8002/docs
2. **Guide de démarrage** : [QUICKSTART.md](QUICKSTART.md)
3. **Guide des stratégies** : [STRATEGIES_GUIDE.md](STRATEGIES_GUIDE.md)
4. **Script de test** : `python api/api_example_client.py`

### Vérifications
1. Logs du serveur (terminal où uvicorn tourne)
2. Statut des jobs : `/api/train/jobs` ou `/api/simulate/jobs`
3. Statut spécifique : `/api/train/{job_id}/status` ou `/api/simulate/{sim_id}/status`

## 🎯 Cas d'Usage

### Trading Court Terme
```json
{
  "strategy": "aggressive",
  "buy_threshold": 0.3,
  "max_loss_percentage": 3.0
}
```

### Trading Moyen Terme
```json
{
  "strategy": "percentage",
  "buy_threshold": 1.5,
  "sell_threshold": 1.5
}
```

### Investissement Long Terme
```json
{
  "strategy": "conservative",
  "min_profit_percentage": 8.0,
  "buy_threshold": 3.0
}
```

---

**Version:** 2.0  
**Dernière mise à jour:** 2025-05-20  
**Auteur:** CAC40 Stock Prediction Team
