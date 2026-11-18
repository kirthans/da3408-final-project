# Guess the Elo: user manual

**What it does:** you give it one chess game, and it estimates how strong each player is, as a Lichess-style rating with a range. It also shows which moves most affected the estimate.

No chess-engine or programming knowledge is needed. The same guide is built into the app under **How it works ↗** (top right).

---

## 1. Open the app

Open **http://localhost:8080** in a browser (or **https://localhost:8443** if HTTPS has been set up). Starting the app itself (`make up`) is covered in the README and is normally done by whoever runs the system.

![The start screen](img/um_1_start.png)

The app has two tabs:
- **Estimate**: analyse a game.
- **Pipeline**: see the model behind the estimates and how it is kept up to date. This is mostly for curious users and administrators.

## 2. Get a game (PGN)

A **PGN** is a short text file that records a chess game: the moves, and often the players, the time control and the clock times.

- **Lichess:** open a finished game, then use the share or export option and download or copy the PGN. Include clock times if offered.
- **Chess.com:** open a finished game, then use Share or Download to copy the PGN.
- **No game to hand?** Click **Online game · clocks** (a real blitz game from the public Lichess database) or **Morphy's Opera game · no clocks** (a famous game from 1858).

## 3. Get an estimate

1. **Paste** the PGN into the box, **drop** a `.pgn` file onto it, or click **Choose a PGN file**. Files can be up to 1 MB.
2. Click **Read game**. If the file holds several games, pick one from **Game to analyse**.
3. If the game has no time control (most over-the-board games), choose **Classical**, **Rapid** or **Blitz**.
4. Click **Estimate ratings**. The computer analyses every move, which usually takes about a second. **Cancel** stops waiting.

![A finished estimate](img/um_2_results.png)

## 4. Read the result

**Rating cards.** For each player:
- The **large number** is the estimated rating, on the Lichess scale for that time control.
- The **green bar** is the **80% range**. Across many games, the true rating falls inside this range about 8 times out of 10, so it is a range, not a promise.
- If the file already contained ratings, a **dashed blue marker** shows them for comparison. **They are never used to make the estimate.**

**Follow the game.** Step through the moves with the arrow buttons, the keyboard arrows (click the board first), or by clicking a move. Symbols mark weaker moves:
- `?!` inaccuracy;
- `?` mistake;
- `??` blunder.

**Moves that shaped the estimate.** These are the moves that pushed the estimate up or down the most. "−87 Elo to Black's estimate" means the estimate for Black would be 87 points higher without that move. Click a move to see its position. Opening moves are not listed: they are often memorised theory, so they say little about a player's own strength.

## 5. What the numbers can't tell you

- **One game is a small sample.** A bad day, an opening the player knows well, or a very short game can move the estimate a lot.
- **The scales differ.** FIDE, national federation and Chess.com ratings use different scales from Lichess.
- **Unsupported games:** bullet, correspondence and chess variants.
- **Without clock times** (most over-the-board games), the estimate has less to go on.
- It **cannot** detect cheating, and it says nothing about a player's potential.

## 6. If something goes wrong

| Message (as shown in the app) | What to do |
|---|---|
| "We could not read this PGN. Export it again with the moves included." | Check that you pasted the whole PGN, including the moves |
| "There are too few moves for a useful estimate. Try a longer game." | Each player needs at least 10 moves. Use a longer game |
| "This file is too large. Choose a PGN up to 1 MB." | Use a smaller file, or paste only the game you want |
| "Bullet and correspondence games aren't supported…" | Use a blitz, rapid or classical game |
| "The rating model is still warming up or unavailable. Please retry shortly." | The system is starting or switching to a new model. Wait a minute and retry |
| "Analysis took too long…" / "The chess engine could not analyse this game…" | Retry, or choose a shorter game |
| "Could not reach the service. Check your connection and retry." | The app's server is not running, or the network is down. Contact the administrator |

Every error shows a **Request ID**. Give it to the administrator so they can find the problem in the logs.

## 7. The Pipeline tab

This tab shows the model and how it stays accurate as player behaviour changes.

![Pipeline tab](img/um_3_pipeline.png)

- **Current model.** The model version in use and its average error on held-out games, in rating points (lower is better). There are links to the experiment records (MLflow) and the monitoring charts (Grafana).
- **2020 replay.** This demonstrates real-world change. The system replays the games of 2020 one month at a time, the year when lockdowns brought many new players to online chess. For the current month it shows:
  - the model's error compared with its usual level (a ratio above 1 means more error than usual);
  - how often the 80% ranges contained the true ratings;
  - whether **drift** was detected.
- **Training steps.** A diagram of the steps from raw games to a trained model. Steps repeated for every month are grouped together. Choose a run to colour its steps by outcome.
- **Run history.** Every time the model was retrained:
  - **promoted** means the new model was better and replaced the old one;
  - **rejected** means it was not better, so the old model stayed;
  - **no change** means nothing needed retraining;
  - **baseline restored** means the replay was reset to the starting model.

  The **Gate** column says whether the new model was better, and by how many rating points. Open **Details** for timings and any error.
- **Request a retrain** (administrators only) needs the API key. The key is not stored by the page.

On a phone, the history turns into cards and the diagram scrolls sideways:

![Pipeline on a phone](img/um_4_pipeline_phone.png)

### Why the model retrains

When the games people play start to differ from the games the model learned from, the model's error grows. The monitoring system watches that error every month. When it stays above 1.04 times the usual level for two minutes, an alert fires automatically. The system then retrains on the newer months, and the new model is only used if it beats the old one on games neither has seen.

## 8. Privacy

- Games are sent to the analysis service to be estimated and are **not stored**.
- PGNs can contain real names. Remove anything you don't want to send.
- Ratings already in the file are only displayed, never used for the estimate.
- **Demo data** mode (open `http://localhost:8080/?mock=1`) shows a built-in example result and is clearly labelled. Nothing is sent or analysed.
