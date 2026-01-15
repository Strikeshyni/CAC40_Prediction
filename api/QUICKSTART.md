# 🚀 Guide de Démarrage Rapide - API Trading CAC40

## Installation et Lancement

```bash
# 1. Installer les dépendances
pip install -r api/requirements_api.txt

# 2. Démarrer l'API
cd /home/abel/personnal_projects/CAC40_stock_prediction
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8002

# 3. Accéder à la documentation interactive
# http://localhost:8002/docs
```

## 🎯 Nouveautés v2.0

### 5 Stratégies de Trading
- **Simple** : Stratégie de base (achat si prédit > actuel)
- **Threshold** : Trade uniquement si la différence dépasse un seuil
- **Percentage** : Basée sur le pourcentage de changement
- **Conservative** : Attend un profit cible avant de vendre
- **Aggressive** : Trade agressif avec stop-loss

### Suivi Détaillé des Simulations
- ✅ Statut en temps réel via WebSocket
- ✅ Historique complet des transactions
- ✅ Analyse détaillée : prix, quantités, raisons d'achat/vente
- ✅ Métriques de performance : win rate, profit/loss

### Nouveaux Endpoints
```
POST   /api/simulate                    # Lancer une simulation
GET    /api/simulate/{sim_id}/status    # Statut de la simulation
GET    /api/simulate/{sim_id}/transactions  # Historique des transactions
GET    /api/simulate/{sim_id}/results   # Résultats complets
GET    /api/simulate/jobs                # Liste de toutes les simulations
DELETE /api/simulate/{sim_id}           # Supprimer une simulation
WS     /ws/simulation/{sim_id}          # WebSocket pour suivi en temps réel
```

## 📖 Exemples Rapides

### 1. Simulation Simple (30 secondes)

```python
import requests

API_URL = "http://localhost:8002"

# Lancer une simulation avec stratégie simple
config = {
    "stock_name": "ENGI.PA",
    "from_date": "2024-11-01",
    "to_date": "2024-11-20",
    "initial_balance": 100.0,
    "strategy": "simple"
}

response = requests.post(f"{API_URL}/api/simulate", json=config)
sim_id = response.json()["sim_id"]

# Vérifier le statut
status = requests.get(f"{API_URL}/api/simulate/{sim_id}/status").json()
print(f"Progression: {status['progress']*100:.0f}%")
```

### 2. Stratégie avec Seuils

```python
# Stratégie threshold : n'achète que si la différence > 1€
config = {
    "stock_name": "ENGI.PA",
    "from_date": "2024-10-01",
    "to_date": "2024-11-20",
    "initial_balance": 100.0,
    "strategy": "threshold",
    "buy_threshold": 1.0,   # Acheter si prédit 1€ au-dessus
    "sell_threshold": 0.8   # Vendre si prédit 0.8€ en-dessous
}

response = requests.post(f"{API_URL}/api/simulate", json=config)
```

### 3. Stratégie Conservative (Long Terme)

```python
# Ne vend que si profit >= 5%
config = {
    "stock_name": "ENGI.PA",
    "from_date": "2024-01-01",
    "to_date": "2024-11-20",
    "initial_balance": 100.0,
    "strategy": "conservative",
    "min_profit_percentage": 5.0,  # Profit cible: 5%
    "buy_threshold": 2.0           # N'achète que si +2% prédit
}

response = requests.post(f"{API_URL}/api/simulate", json=config)
```

### 4. Stratégie Aggressive avec Stop-Loss

```python
# Trade agressif avec protection stop-loss
config = {
    "stock_name": "ENGI.PA",
    "from_date": "2024-06-01",
    "to_date": "2024-11-20",
    "initial_balance": 100.0,
    "strategy": "aggressive",
    "buy_threshold": 0.3,          # Très sensible: achète si +0.3%
    "max_loss_percentage": 3.0     # Stop-loss à -3%
}

response = requests.post(f"{API_URL}/api/simulate", json=config)
```

### 5. Récupérer les Transactions

```python
# Attendre que la simulation soit terminée
import time

while True:
    status = requests.get(f"{API_URL}/api/simulate/{sim_id}/status").json()
    if status["status"] == "completed":
        break
    time.sleep(2)

# Récupérer toutes les transactions
transactions = requests.get(f"{API_URL}/api/simulate/{sim_id}/transactions").json()

print(f"Total transactions: {transactions['total_transactions']}")
for t in transactions['transactions'][-5:]:  # 5 dernières
    print(f"{t['date']} | {t['transaction_type'].upper()} | "
          f"{t['quantity']:.2f} @ {t['stock_price']:.2f}€")
    print(f"  Raison: {t['reason']}")
```

### 6. Suivi en Temps Réel (WebSocket)

```javascript
const ws = new WebSocket('ws://localhost:8002/ws/simulation/{sim_id}');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(`Progression: ${(data.progress * 100).toFixed(1)}%`);
  console.log(`Date: ${data.current_date}`);
  console.log(`Balance: ${data.current_balance.toFixed(2)}€`);
  console.log(`Transactions: ${data.total_transactions}`);
  
  if (data.status === 'completed') {
    console.log('Simulation terminée!');
    ws.close();
  }
};
```

## 📊 Script de Test Complet

Un script de test complet est disponible :

```bash
python api/api_example_client.py
```

Ce script teste :
- ✅ Connexion à l'API
- ✅ Simulation simple
- ✅ Comparaison des 5 stratégies
- ✅ Affichage des résultats

## 📚 Documentation Complète

- **[README_API.md](README_API.md)** : Documentation complète de tous les endpoints
- **[STRATEGIES_GUIDE.md](STRATEGIES_GUIDE.md)** : Guide détaillé des stratégies avec exemples
- **[http://localhost:8002/docs](http://localhost:8002/docs)** : Documentation interactive Swagger

## 🔧 Endpoints Principaux

### Simulations
| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/api/simulate` | POST | Lancer une simulation |
| `/api/simulate/{sim_id}/status` | GET | Statut de la simulation |
| `/api/simulate/{sim_id}/transactions` | GET | Liste des transactions |
| `/api/simulate/{sim_id}/results` | GET | Résultats complets |
| `/api/simulate/jobs` | GET | Toutes les simulations |

### Entraînement
| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/api/train` | POST | Entraîner un modèle |
| `/api/train/{job_id}/status` | GET | Statut de l'entraînement |
| `/api/train/jobs` | GET | Tous les entraînements |

### Prédictions
| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/api/predict` | POST | Faire des prédictions |

## 💡 Conseils d'Utilisation

### Pour Débuter
1. Testez d'abord la **stratégie "simple"** sur une **courte période** (1 mois)
2. Utilisez le **script de test** : `python api/api_example_client.py`
3. Consultez la **documentation interactive** : http://localhost:8002/docs

### Pour Optimiser
1. **Comparez les stratégies** sur la même période
2. **Ajustez les seuils** en fonction de la volatilité du stock
3. **Analysez les transactions** pour comprendre les décisions

### Pour des Simulations Longues
1. Utilisez **WebSocket** pour suivre la progression
2. Vérifiez régulièrement le **statut** : `/api/simulate/{sim_id}/status`
3. Les modèles sont **mis en cache** - les simulations suivantes seront plus rapides

## ⚙️ Paramètres des Stratégies

| Stratégie | Paramètres | Valeurs Recommandées |
|-----------|-----------|---------------------|
| **Simple** | Aucun | - |
| **Threshold** | `buy_threshold`, `sell_threshold` | 0.5€ - 2.0€ |
| **Percentage** | `buy_threshold`, `sell_threshold` | 1.0% - 3.0% |
| **Conservative** | `min_profit_percentage`, `buy_threshold` | 5% - 10%, 2% |
| **Aggressive** | `max_loss_percentage`, `buy_threshold` | 3% - 5%, 0.3% |

## 🚨 Limitations

- Les simulations ne prennent **pas en compte les frais de transaction**
- Le backtesting **ne garantit pas** les performances futures
- Les simulations longues (>6 mois) peuvent **prendre du temps**
- Attendez quelques heures en cas d'erreur **"Rate Limit"** de Yahoo Finance

## 🆘 Support & Debugging

### L'API ne démarre pas
```bash
# Vérifier les logs
python -m uvicorn api.main:app --log-level debug --port 8002
```

### Simulation bloquée
```bash
# Vérifier le statut
curl http://localhost:8002/api/simulate/{sim_id}/status

# Voir les erreurs dans les logs du serveur
```

### Comparer les performances
```bash
# Utiliser le script de test
python api/api_example_client.py
```

## 📈 Exemple de Résultat

```
╔════════════════════════════════════════════════════════════════════════════════╗
║                    COMPARAISON DES STRATÉGIES                                   ║
╚════════════════════════════════════════════════════════════════════════════════╝

Stratégie                 | Balance    | Profit        | Trades   | Win Rate   
────────────────────────────────────────────────────────────────────────────────
Aggressive (5%)           |   115.23€ | +15.23% (+15.23€) |     45 |    68.2%
Conservative (3%)         |   112.45€ | +12.45% (+12.45€) |     12 |    83.3%
Percentage (1.5%)         |   108.67€ |  +8.67%  (+8.67€) |     28 |    64.3%
Simple                    |   106.34€ |  +6.34%  (+6.34€) |     34 |    55.9%
Threshold (0.5€)          |   103.12€ |  +3.12%  (+3.12€) |     18 |    61.1%
────────────────────────────────────────────────────────────────────────────────
```

---

**Version:** 2.0  
**Dernière mise à jour:** 2025-05-20
