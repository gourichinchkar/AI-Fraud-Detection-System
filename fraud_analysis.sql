
CREATE DATABASE fraud_detection;

USE fraud_detection;
SELECT COUNT(*) FROM transactions;
USE fraud_detection;
SELECT * FROM cleaned_fraudTest;

SELECT COUNT(*) AS Total_Fraud_Transactions
FROM cleaned_fraudTest
WHERE is_fraud = 1;

-- Highest Fraud Transaction Amount--  
SELECT MAX(amt) AS Highest_Fraud_Amount
FROM cleaned_fraudTest
WHERE is_fraud = 1;

-- Location-wise Fraud Analysis--  
SELECT state, COUNT(*) AS Total_Fraud
FROM cleaned_fraudTest
WHERE is_fraud = 1
GROUP BY state
ORDER BY Total_Fraud DESC;

-- Customer Transaction Trends-- 
SELECT cc_num, COUNT(*) AS Total_Fraud
FROM cleaned_fraudTest
WHERE is_fraud = 1
GROUP BY cc_num
ORDER BY Total_Fraud DESC
LIMIT 10;

-- Merchant Risk Analysis--
SELECT merchant, COUNT(*) AS Total_Fraud
FROM cleaned_fraudTest
WHERE is_fraud = 1
GROUP BY merchant
ORDER BY Total_Fraud DESC
LIMIT 10;