"""
Script d'exemple pour tester l'API de simulation de trading

Utilisation:
    python api_example_client.py
"""

import requests
import time
import json
from datetime import datetime, timedelta

API_URL = "http://localhost:8002"


def test_training():
    """Test de l'entraînement d'un modèle"""
    print("\n" + "="*80)
    print("TEST 1: ENTRAÎNEMENT D'UN MODÈLE")
    print("="*80)
    
    config = {
        "stock_name": "ENGI.PA",
        "from_date": "2023-01-01",
        "to_date": "2024-12-31",
        "train_size_percent": 0.8,
        "val_size_percent": 0.2,
        "time_step": 300,
        "global_tuning": False  # Plus rapide pour le test
    }
    
    response = requests.post(f"{API_URL}/api/train", json=config)
    result = response.json()
    job_id = result["job_id"]
    
    print(f"✓ Job créé: {job_id}")
    print(f"  Status: {result['status']}")
    
    # Suivre la progression
    print("\nProgression:")
    while True:
        status_response = requests.get(f"{API_URL}/api/train/{job_id}/status")
        status = status_response.json()
        
        print(f"  [{status['progress']*100:5.1f}%] {status['status']:10s} - {status['current_step']}")
        
        if status['status'] in ['completed', 'failed']:
            break
        
        time.sleep(2)
    
    if status['status'] == 'completed':
        print(f"\n✓ Entraînement terminé! Modèle sauvegardé: {status['model_path']}")
        return job_id
    else:
        print(f"\n✗ Erreur: {status.get('error', 'Unknown error')}")
        return None


def test_prediction(job_id):
    """Test des prédictions"""
    print("\n" + "="*80)
    print("TEST 2: PRÉDICTIONS")
    print("="*80)
    
    pred_request = {
        "job_id": job_id,
        "n_days": 7
    }
    
    response = requests.post(f"{API_URL}/api/predict", json=pred_request)
    predictions = response.json()
    
    print(f"\nStock: {predictions['stock_name']}")
    print(f"Dernier prix actuel: {predictions['last_actual_price']:.2f}€ ({predictions['last_actual_date']})")
    print(f"\nPrédictions pour les 7 prochains jours:")
    
    for pred in predictions['predictions']:
        change = pred['predicted_price'] - predictions['last_actual_price']
        change_pct = (change / predictions['last_actual_price']) * 100
        arrow = "↗" if change > 0 else "↘" if change < 0 else "→"
        print(f"  Jour {pred['day']}: {pred['predicted_price']:6.2f}€  {arrow} {change_pct:+.2f}%")


def test_simple_simulation():
    """Test d'une simulation simple"""
    print("\n" + "="*80)
    print("TEST 3: SIMULATION SIMPLE (Stratégie basique)")
    print("="*80)
    
    config = {
        "stock_name": "ENGI.PA",
        "from_date": (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
        "to_date": datetime.now().strftime('%Y-%m-%d'),
        "initial_balance": 100.0,
        "time_step": 300,
        "nb_years_data": 5,
        "strategy": "simple"
    }
    
    print(f"Période: {config['from_date']} → {config['to_date']}")
    print(f"Capital initial: {config['initial_balance']}€")
    
    response = requests.post(f"{API_URL}/api/simulate", json=config)
    result = response.json()
    sim_id = result["sim_id"]
    
    print(f"✓ Simulation créée: {sim_id}")
    
    # Suivre la progression
    print("\nProgression:")
    while True:
        status_response = requests.get(f"{API_URL}/api/simulate/{sim_id}/status")
        status = status_response.json()
        
        progress_bar = "█" * int(status['progress'] * 30) + "░" * (30 - int(status['progress'] * 30))
        print(f"\r  [{progress_bar}] {status['progress']*100:5.1f}% | "
              f"Date: {status.get('current_date', 'N/A'):10s} | "
              f"Balance: {status['current_balance']:7.2f}€ | "
              f"Trades: {status['total_transactions']:3d}", end='')
        
        if status['status'] in ['completed', 'failed']:
            print()  # Nouvelle ligne
            break
        
        time.sleep(1)
    
    if status['status'] == 'completed':
        # Récupérer les résultats complets
        results = requests.get(f"{API_URL}/api/simulate/{sim_id}/results").json()
        summary = results['summary']
        
        print(f"\n{'─'*80}")
        print("RÉSULTATS:")
        print(f"  Balance finale: {results['final_balance']:.2f}€")
        print(f"  Profit/Perte:   {results['benefit']:+.2f}€ ({results['benefit_percentage']:+.2f}%)")
        print(f"  Trades totaux:  {summary['total_trades']}")
        print(f"  Achats:         {summary['buy_trades']}")
        print(f"  Ventes:         {summary['sell_trades']}")
        print(f"  Taux de gain:   {summary['win_rate']:.1f}%")
        
        # Afficher quelques transactions
        transactions_response = requests.get(f"{API_URL}/api/simulate/{sim_id}/transactions")
        transactions = transactions_response.json()['transactions']
        
        if transactions:
            print(f"\nDernières transactions:")
            for t in transactions[-5:]:
                print(f"  {t['date']} | {t['transaction_type'].upper():4s} | "
                      f"{t['quantity']:6.2f} @ {t['stock_price']:6.2f}€ | {t['reason']}")
        
        return sim_id
    else:
        print(f"\n✗ Erreur: {status.get('error', 'Unknown error')}")
        return None


def test_strategies_comparison():
    """Comparer plusieurs stratégies"""
    print("\n" + "="*80)
    print("TEST 4: COMPARAISON DES STRATÉGIES")
    print("="*80)
    
    # Période de test courte pour aller plus vite
    from_date = (datetime.now() - timedelta(days=20)).strftime('%Y-%m-%d')
    to_date = datetime.now().strftime('%Y-%m-%d')
    
    strategies = [
        {
            "name": "Simple",
            "config": {"strategy": "simple"}
        },
        {
            "name": "Threshold (0.5€)",
            "config": {"strategy": "threshold", "buy_threshold": 0.5, "sell_threshold": 0.5}
        },
        {
            "name": "Percentage (1.5%)",
            "config": {"strategy": "percentage", "buy_threshold": 1.5, "sell_threshold": 1.5}
        },
        {
            "name": "Conservative (3%)",
            "config": {"strategy": "conservative", "min_profit_percentage": 3.0}
        },
        {
            "name": "Aggressive (5%)",
            "config": {"strategy": "aggressive", "max_loss_percentage": 5.0}
        }
    ]
    
    results = []
    
    for strat in strategies:
        config = {
            "stock_name": "ENGI.PA",
            "from_date": from_date,
            "to_date": to_date,
            "initial_balance": 100.0,
            "time_step": 300,
            "nb_years_data": 5,
            **strat["config"]
        }
        
        print(f"\n{strat['name']:25s} ", end='')
        
        response = requests.post(f"{API_URL}/api/simulate", json=config)
        sim_id = response.json()["sim_id"]
        
        # Attendre la fin
        while True:
            status = requests.get(f"{API_URL}/api/simulate/{sim_id}/status").json()
            print(".", end='', flush=True)
            
            if status["status"] in ["completed", "failed"]:
                break
            time.sleep(2)
        
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
            print(f" ✓ {result['benefit_percentage']:+6.2f}%")
        else:
            print(f" ✗ Failed")
    
    # Afficher tableau comparatif
    print(f"\n{'─'*100}")
    print(f"{'Stratégie':<25s} | {'Balance':<10s} | {'Profit':<12s} | {'Trades':<8s} | {'Win Rate':<10s}")
    print(f"{'─'*100}")
    
    for r in sorted(results, key=lambda x: x['benefit_pct'], reverse=True):
        print(f"{r['strategy']:<25s} | {r['final_balance']:8.2f}€ | "
              f"{r['benefit_pct']:+6.2f}% ({r['benefit']:+6.2f}€) | "
              f"{r['trades']:6d} | {r['win_rate']:7.1f}%")
    
    print(f"{'─'*100}")


def main():
    """Fonction principale"""
    print("\n")
    print("╔" + "═"*78 + "╗")
    print("║" + " "*20 + "API DE TRADING - TESTS COMPLETS" + " "*27 + "║")
    print("╚" + "═"*78 + "╝")
    
    try:
        # Test 1: Vérifier que l'API est accessible
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            print(f"\n✓ API accessible sur {API_URL}")
        else:
            print(f"\n✗ API retourne status code {response.status_code}")
            return
    except requests.exceptions.RequestException as e:
        print(f"\n✗ Impossible de contacter l'API sur {API_URL}")
        print(f"  Erreur: {e}")
        print(f"\n  Assurez-vous que l'API est démarrée:")
        print(f"  python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8002")
        return
    
    # Lancer les tests
    try:
        # job_id = test_training()
        # if job_id:
        #     test_prediction(job_id)
        
        test_simple_simulation()
        test_strategies_comparison()
        
        print("\n" + "="*80)
        print("✓ Tous les tests terminés!")
        print("="*80 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n✗ Tests interrompus par l'utilisateur")
    except Exception as e:
        print(f"\n\n✗ Erreur lors des tests: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
