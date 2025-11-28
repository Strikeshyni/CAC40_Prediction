# Guide des Stratégies de Trading

## 📊 Stratégies Disponibles

L'API propose 5 stratégies de trading différentes, chacune avec ses propres paramètres et comportements.

### 1. Simple Strategy (`"simple"`)

**Description** : Stratégie de base qui achète si le prix prédit est supérieur au prix actuel, et vend dans le cas inverse.

**Paramètres** : Aucun

**Comportement** :
- **Achat** : Si `predicted_price > actual_price` et il reste du cash
- **Vente** : Si `predicted_price < actual_price` et on possède des actions

**Exemple** :
```json
{
  "stock_name": "ENGI.PA",
  "from_date": "2024-01-01",
  "to_date": "2024-12-31",
  "initial_balance": 100.0,
  "strategy": "simple"
}
```

### 2. Threshold Strategy (`"threshold"`)

**Description** : Ne trade que si la différence de prix dépasse un seuil (en euros).

**Paramètres** :
- `buy_threshold` : Différence minimale pour acheter (défaut: 0.5€)
- `sell_threshold` : Différence minimale pour vendre (défaut: 0.5€)

**Comportement** :
- **Achat** : Si `predicted_price - actual_price > buy_threshold`
- **Vente** : Si `actual_price - predicted_price > sell_threshold`

**Exemple** :
```json
{
  "stock_name": "ENGI.PA",
  "from_date": "2024-01-01",
  "to_date": "2024-12-31",
  "initial_balance": 100.0,
  "strategy": "threshold",
  "buy_threshold": 1.0,
  "sell_threshold": 0.8
}
```

**Cas d'usage** : Éviter les trades sur des petites fluctuations, réduire les frais de transaction.

### 3. Percentage Strategy (`"percentage"`)

**Description** : Trade basé sur le pourcentage de changement prédit.

**Paramètres** :
- `buy_threshold` : Pourcentage minimum de hausse prédite pour acheter (défaut: 1.0%)
- `sell_threshold` : Pourcentage minimum de baisse prédite pour vendre (défaut: 1.0%)

**Comportement** :
- **Achat** : Si `((predicted - actual) / actual) * 100 > buy_threshold`
- **Vente** : Si `((predicted - actual) / actual) * 100 < -sell_threshold`

**Exemple** :
```json
{
  "stock_name": "ENGI.PA",
  "from_date": "2024-01-01",
  "to_date": "2024-12-31",
  "initial_balance": 100.0,
  "strategy": "percentage",
  "buy_threshold": 2.0,
  "sell_threshold": 1.5
}
```

**Cas d'usage** : S'adapter aux actions de différentes valeurs, trader proportionnellement.

### 4. Conservative Strategy (`"conservative"`)

**Description** : Stratégie prudente qui attend un profit cible avant de vendre.

**Paramètres** :
- `min_profit_percentage` : Profit minimum avant vente (défaut: 5.0%)
- `buy_threshold` : Pourcentage de hausse prédite pour acheter (défaut: 2.0%)

**Comportement** :
- **Achat** : Si hausse prédite > buy_threshold (défaut 2%)
- **Vente** : 
  - Si profit actuel >= min_profit_percentage, OU
  - Si baisse prédite > 1%

**Exemple** :
```json
{
  "stock_name": "ENGI.PA",
  "from_date": "2024-01-01",
  "to_date": "2024-12-31",
  "initial_balance": 100.0,
  "strategy": "conservative",
  "min_profit_percentage": 10.0,
  "buy_threshold": 3.0
}
```

**Cas d'usage** : Investissement à long terme, minimiser les pertes, attendre de bons profits.

### 5. Aggressive Strategy (`"aggressive"`)

**Description** : Trade agressif sur de petits signaux avec stop-loss.

**Paramètres** :
- `buy_threshold` : Pourcentage minimum de hausse prédite (défaut: 0.1%)
- `max_loss_percentage` : Perte maximale avant stop-loss (défaut: 5.0%)

**Comportement** :
- **Achat** : Si hausse prédite > 0.1% (très sensible)
- **Vente** :
  - Si perte actuelle >= max_loss_percentage (STOP-LOSS), OU
  - Si baisse prédite (n'importe quelle baisse)

**Exemple** :
```json
{
  "stock_name": "ENGI.PA",
  "from_date": "2024-01-01",
  "to_date": "2024-12-31",
  "initial_balance": 100.0,
  "strategy": "aggressive",
  "buy_threshold": 0.5,
  "max_loss_percentage": 3.0
}
```

**Cas d'usage** : Trading court terme, maximiser le nombre de trades, protection par stop-loss.

## 📈 Comparaison des Stratégies

| Stratégie | Fréquence de trading | Risque | Profit potentiel | Meilleur pour |
|-----------|---------------------|--------|------------------|---------------|
| Simple | Moyen | Moyen | Moyen | Débutants, tests |
| Threshold | Faible | Faible | Faible-Moyen | Réduire les frais |
| Percentage | Moyen | Moyen | Moyen | Actions variées |
| Conservative | Faible | Faible | Moyen-Élevé | Long terme |
| Aggressive | Élevé | Élevé | Élevé | Court terme |

## 🎯 Exemples Complets

### Test de Plusieurs Stratégies

```python
import requests
import time

API_URL = "http://localhost:8002"

strategies = [
    {"name": "Simple", "config": {"strategy": "simple"}},
    {"name": "Threshold 1€", "config": {"strategy": "threshold", "buy_threshold": 1.0, "sell_threshold": 1.0}},
    {"name": "Percentage 2%", "config": {"strategy": "percentage", "buy_threshold": 2.0, "sell_threshold": 2.0}},
    {"name": "Conservative 5%", "config": {"strategy": "conservative", "min_profit_percentage": 5.0}},
    {"name": "Aggressive 3%", "config": {"strategy": "aggressive", "max_loss_percentage": 3.0}},
]

results = []

for strat in strategies:
    config = {
        "stock_name": "ENGI.PA",
        "from_date": "2024-01-01",
        "to_date": "2024-03-31",
        "initial_balance": 100.0,
        "time_step": 300,
        "nb_years_data": 10,
        **strat["config"]
    }
    
    # Lancer la simulation
    response = requests.post(f"{API_URL}/api/simulate", json=config)
    sim_id = response.json()["sim_id"]
    print(f"Testing {strat['name']}... (sim_id: {sim_id})")
    
    # Attendre la fin
    while True:
        status = requests.get(f"{API_URL}/api/simulate/{sim_id}/status").json()
        if status["status"] in ["completed", "failed"]:
            break
        time.sleep(5)
    
    # Récupérer les résultats
    if status["status"] == "completed":
        result = requests.get(f"{API_URL}/api/simulate/{sim_id}/results").json()
        results.append({
            "strategy": strat["name"],
            "final_balance": result["final_balance"],
            "benefit": result["benefit"],
            "benefit_pct": result["benefit_percentage"],
            "trades": result["summary"]["total_trades"],
            "win_rate": result["summary"]["win_rate"]
        })

# Afficher les résultats
print("\n" + "="*80)
print("COMPARAISON DES STRATÉGIES")
print("="*80)
for r in results:
    print(f"\n{r['strategy']:20s} | Balance finale: {r['final_balance']:7.2f}€ | "
          f"Profit: {r['benefit_pct']:+6.2f}% | Trades: {r['trades']:3d} | "
          f"Win rate: {r['win_rate']:5.1f}%")
```

### Optimisation des Paramètres

```python
# Tester différents seuils pour la stratégie threshold
thresholds = [0.5, 1.0, 1.5, 2.0, 2.5]

for threshold in thresholds:
    config = {
        "stock_name": "ENGI.PA",
        "from_date": "2024-01-01",
        "to_date": "2024-12-31",
        "initial_balance": 100.0,
        "strategy": "threshold",
        "buy_threshold": threshold,
        "sell_threshold": threshold
    }
    
    response = requests.post(f"{API_URL}/api/simulate", json=config)
    sim_id = response.json()["sim_id"]
    
    # ... attendre et récupérer résultats
```

## 📊 Suivi en Temps Réel

### WebSocket pour Progression

```javascript
const ws = new WebSocket('ws://localhost:8002/ws/simulation/{sim_id}');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(`Progress: ${(data.progress * 100).toFixed(1)}%`);
  console.log(`Current date: ${data.current_date}`);
  console.log(`Balance: ${data.current_balance.toFixed(2)}€`);
  console.log(`Stocks: ${data.current_stocks.toFixed(4)}`);
  console.log(`Transactions: ${data.total_transactions}`);
};
```

### Récupération des Transactions

```python
# Récupérer toutes les transactions
response = requests.get(f"{API_URL}/api/simulate/{sim_id}/transactions")
transactions = response.json()

print(f"Total transactions: {transactions['total_transactions']}")
print(f"\nDernières transactions:")
for t in transactions['transactions'][-5:]:
    print(f"{t['date']} - {t['transaction_type'].upper():4s} - "
          f"{t['quantity']:.2f} actions @ {t['stock_price']:.2f}€ - "
          f"Raison: {t['reason']}")
```

## 💡 Conseils d'Utilisation

1. **Commencez par "simple"** pour comprendre le comportement de base
2. **Testez sur des périodes courtes** avant de lancer des simulations longues
3. **Comparez plusieurs stratégies** sur la même période
4. **Ajustez les seuils** en fonction de la volatilité de l'action
5. **Analysez les transactions** pour comprendre les décisions prises
6. **Utilisez WebSocket** pour suivre les longues simulations en temps réel

## ⚠️ Limitations

- Les simulations ne prennent pas en compte les frais de transaction
- Les prédictions peuvent être imprécises en période de forte volatilité
- Le backtesting ne garantit pas les performances futures
- Les stratégies sont conçues pour le day-trading/swing-trading, pas l'investissement à long terme
