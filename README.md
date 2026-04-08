# March Machine Learning Mania 2026 — 3rd Place Solution

**Final Score:** 0.1160374 (MSE/Brier, 126 games) | **Rank:** 3rd / 3,485 teams

## Approach

Logistic Regression on team-level differential features + triple-market probability blend (ESPN BPI, Vegas moneylines, Kalshi prediction markets). Separate men's/women's models. 52 rounds of iteration.

See [SOLUTION_WRITEUP.md](SOLUTION_WRITEUP.md) for the full write-up.

## Reproduction

### Requirements

```
Python 3.9+
numpy, pandas, scikit-learn, scipy
```

The pipeline also requires official Kaggle competition data:

```bash
pip install kaggle
kaggle competitions download -c march-machine-learning-mania-2026 -p data/
cd data && unzip '*.zip' && rm *.zip && cd ..
```

### Generate Submission

```python
python3 -c "
import round52_final as r52
sub = r52.generate_submission(alpha_m_r1=0.90, alpha_w_r1=0.75)
sub.to_csv('submission.csv', index=False)
print(f'Wrote {len(sub)} rows to submission.csv')
"
```

Key parameters for the winning submission:
- `alpha_m_r1=0.90` — 90% market weight for men's Round 1
- `alpha_w_r1=0.75` — 75% market weight for women's Round 1

### File Structure

```
round27_pruned.py   — Base feature engineering (Elo, Four Factors, Barttorvik, Massey, etc.)
round45_final.py    — Colley Matrix, GLM quality, interactions, all-system Massey
round46_final.py    — Kalshi markets, additional interactions
round49_final.py    — Simple Rating System (SRS)
round50_final.py    — ESPN BPI integration (game predictions + championship odds)
round51_final.py    — Coach PASE, Women's BPI
round52_final.py    — Triple-market blend (ESPN BPI + Vegas + Kalshi), final submission
external/           — External datasets (Barttorvik, KenPom, EvanMiya, AP Poll, etc.)
```

Import chain: `round27 → round45 → round46 → round49 → round50 → round51 → round52`

## Competition

[March Machine Learning Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026) on Kaggle.