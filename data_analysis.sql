
---create database top5_leagues_player;

---USE top5_leagues_player;

---CREATE TABLE top5_leagues_players(
   ---id INT PRIMARY KEY,
    ---name VARCHAR(255),
    ---full_name VARCHAR(255),
    ---age INT,
    ---height FLOAT,
    ---nationality VARCHAR(255),
    ---place_of_birth VARCHAR(255),
    ---price FLOAT,
    ---max_price FLOAT,
    ---position VARCHAR(255),
    ---shirt_nr INT,
    ---foot VARCHAR(50),
    ---club VARCHAR(255),
    ---contract_expires DATE,
    ---joined_club DATE,
    ---player_agent VARCHAR(255),
    ---outfitter VARCHAR(255),
    ---league VARCHAR(50)
---);



---BULK INSERT top5_leagues_players
---FROM 'D:\New folder\New folder (7)\top5_leagues_player.csv'
---WITH (
   --- FORMAT = 'CSV',
    ---FIRSTROW = 2,  -- Skips the header row
    ---FIELDTERMINATOR = ',',  -- Defines comma as column separator
    ---ROWTERMINATOR = '\n',  -- Defines new line as row separator
    ---TABLOCK
---);

---Data cleaning
---Handle Missing values
-- Replace NULL height values with the average height
---UPDATE top5_leagues_players
---SET height = (SELECT AVG(height) FROM top5_leagues_players)
---WHERE height IS NULL;

-- Replace NULL price with 0 (or average price)
---UPDATE top5_leagues_players
---SET price = 0
---WHERE price IS NULL;

-- Standardize missing dates (replace NULL with 'Unknown' for categorical fields)
---UPDATE top5_leagues_players
---SET contract_expires = NULL
---WHERE contract_expires = '';

---UPDATE top5_leagues_players
---SET joined_club = NULL
---WHERE joined_club = '';

---Data Transformation
---Extract player experience(Years at club)
---ALTER TABLE top5_leagues_players ADD years_at_club INT;
---UPDATE top5_leagues_players
---SET years_at_club = DATEDIFF(YEAR, joined_club, GETDATE());


---UPDATE top5_leagues_players
---SET years_at_club = DATEDIFF(YEAR, joined_club, GETDATE())
---WHERE joined_club IS NOT NULL;

---Calculate Market Value Difference
---ALTER TABLE top5_leagues_players ADD price_difference FLOAT;

---UPDATE top5_leagues_players
---SET price_difference = max_price - price;

---Analysis queries
---Top 10 most expensive players
SELECT TOP 10 name, club, price, position, league
FROM top5_leagues_players
ORDER BY price DESC;

---Average Age of Players by Leagues
---SELECT league, AVG(age) AS avg_age
---FROM top5_leagues_players
---GROUP BY league
---ORDER BY avg_age DESC;

---Number of Players Per Position
---SELECT position, COUNT(*) AS player_count
---FROM top5_leagues_players
---GROUP BY position
---ORDER BY player_count DESC;

---top 5 Clubs with Highest Total Market Value
---SELECT TOP 5 club, SUM(price) AS total_value
---FROM top5_leagues_players
---GROUP BY club
---ORDER BY total_value DESC;

---Players with Highest Market Value Growth
---SELECT TOP 10 name, club, price, max_price, price_difference
---FROM top5_leagues_players
---ORDER BY price_difference DESC;

