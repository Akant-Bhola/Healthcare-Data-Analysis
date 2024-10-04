# Databricks notebook source
# MAGIC %md
# MAGIC # **Healthcare Data Analysis**
# MAGIC
# MAGIC **Introduction:**
# MAGIC
# MAGIC The purpose of this project is to analyse patient data from a healthcare dataset to extract meaningful insights using big data techniques and Databricks tools. The analysis focuses on understanding various patterns in patient demographics, medical conditions, hospital admissions and billing amounts to help healthcare organisations make informed decisions. We aim to use the data to identify trends in patient admissions, the frequency of medical conditions and the cost distribution associated with different treatments and hospitals.
# MAGIC
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC **Data Loading and Schema Inspection:**
# MAGIC
# MAGIC The Kaggle healthcare dataset is loaded from DBFS into a PySpark DataFrame. The schema is inspected to understand the structure and data types, ensuring the data is ready for further analysis.
# MAGIC
# MAGIC **Data Transformation:**
# MAGIC
# MAGIC Columns like "Name" are transformed to standardize the formatting (e.g., capitalizing the first letter).
# MAGIC

# COMMAND ----------

df = spark.read.csv("/FileStore/tables/healthcare_dataset.csv", header=True, inferSchema=True)
display(df)

# COMMAND ----------

# Checking the schema of the DataFrame
df.printSchema()

# COMMAND ----------

#Capitalising first letter of Name
from pyspark.sql.functions import initcap

df = df.withColumn("Name", initcap("Name"))
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC **Analysis:**
# MAGIC
# MAGIC **Average Billing Amount by Medical Condition:**
# MAGIC
# MAGIC The lowest average billing is for Diabetes at 25,198, while the highest is for Asthma at 26,086. This suggests that billing costs vary slightly by medical condition, with asthma treatments or procedures generally costing more on average than those for diabetes.
# MAGIC
# MAGIC **Billing Distribution by Insurance Provider:**
# MAGIC
# MAGIC The box plot shows a nearly normal distribution for billing amounts across all insurance providers, with the median values ranging from 25,000 to over 26,000. This indicates a consistent range of billing costs regardless of the insurance provider.
# MAGIC
# MAGIC **Billing Distribution by Month:**
# MAGIC
# MAGIC The trend in billing distribution shows a decline from May 2019 (highest at 3,036,708) to May 2024 (lowest at 912,944). The highest billing is observed in August 2020 at 5,338,718. This suggests significant variability in monthly billing amounts over time, possibly due to external factors such as healthcare demand, policy changes or seasonal variations.

# COMMAND ----------

# MAGIC %md
# MAGIC **Descriptive Analysis:**
# MAGIC
# MAGIC Histograms are generated to analyze the distribution of patient ages and billing amounts.
# MAGIC The frequency of medical conditions is analyzed based on admission type and blood type.
# MAGIC

# COMMAND ----------

# MAGIC %pip install matplotlib pandas numpy
# MAGIC
# MAGIC import matplotlib.pyplot as plt
# MAGIC import pandas as pd
# MAGIC import numpy as np
# MAGIC
# MAGIC
# MAGIC # Adjusting bins to have a wider range for Age Distribution
# MAGIC age_histogram = df.select("Age").rdd.flatMap(lambda x: [x]).histogram(list(np.arange(start=min(df.select("Age").rdd.flatMap(lambda x: [x]).min()), 
# MAGIC                                                                              stop=max(df.select("Age").rdd.flatMap(lambda x: [x]).max()) + 10, 
# MAGIC                                                                              step=5)))
# MAGIC age_pd = pd.DataFrame(list(zip(*age_histogram)), columns=['Age', 'Frequency'])
# MAGIC
# MAGIC # Plotting with frequency legend outside graph area
# MAGIC fig, ax = plt.subplots()
# MAGIC age_pd.plot(kind='bar', x='Age', y='Frequency', title="Age Distribution", ax=ax)
# MAGIC ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# MAGIC plt.close(fig)  # Prevents the automatic display of the figure in Jupyter-like environments
# MAGIC display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC **Analysis:**
# MAGIC
# MAGIC - Most patients are within the age range of 18 to 78, showing a relatively uniform distribution in this age group.
# MAGIC
# MAGIC - The frequency drops significantly for ages below 18 and above 78, indicating fewer admissions in these age groups.
# MAGIC

# COMMAND ----------

# MAGIC %pip install matplotlib pandas
# MAGIC
# MAGIC import matplotlib.pyplot as plt
# MAGIC import pandas as pd
# MAGIC import numpy as np
# MAGIC
# MAGIC # Adjusting bins for Billing Amount Distribution in whole numbers and increasing bin range in thousands
# MAGIC billing_amount_min = df.select("Billing Amount").rdd.flatMap(lambda x: x).min()
# MAGIC billing_amount_max = df.select("Billing Amount").rdd.flatMap(lambda x: x).max()
# MAGIC bin_range = np.arange(start=billing_amount_min, stop=billing_amount_max + 2000, step=2000).tolist()  # Adjust step for bin size
# MAGIC
# MAGIC billing_histogram = df.select("Billing Amount").rdd.flatMap(lambda x: [x[0]]).histogram(bin_range)
# MAGIC billing_pd = pd.DataFrame(list(zip(*billing_histogram)), columns=['Billing Amount', 'Frequency'])
# MAGIC
# MAGIC # Remove decimals from Billing Amount
# MAGIC billing_pd['Billing Amount'] = billing_pd['Billing Amount'].astype(int)
# MAGIC
# MAGIC # Plotting with frequency legend outside graph area
# MAGIC fig, ax = plt.subplots()
# MAGIC billing_pd.plot(kind='bar', x='Billing Amount', y='Frequency', title="Billing Amount Distribution", ax=ax)
# MAGIC ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# MAGIC plt.close(fig)  # Prevents the automatic display of the figure in Jupyter-like environments
# MAGIC display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC **Analysis:**
# MAGIC
# MAGIC - Billing amounts between 1991 and 47,991 have relatively uniform frequencies, indicating a consistent range of treatment costs, with billing amounts up to 45991 have a consistently high frequency, i.e., above 2000
# MAGIC
# MAGIC
# MAGIC - Extremely low (< 1991) or high billing amounts (> 47991) show significantly lower frequencies, suggesting these are less common. Negative billing amounts are retained in the dataset as they may represent refunds, adjustments or corrections in the billing records. 

# COMMAND ----------

#Creating a graph of Medical Condition by Admission Type 
from pyspark.sql.functions import count

# Grouping data by Medical Condition and Admission Type, then counting occurrences
histogram_data = df.groupBy("Medical Condition", "Admission Type").agg(count("*").alias("Frequency"))

display(histogram_data)

# COMMAND ----------

# MAGIC %md
# MAGIC **Analysis:**
# MAGIC
# MAGIC The nearly uniform distribution of chronic conditions across elective, urgent, and emergency care suggests these issues are consistently critical regardless of care type, necessitating comprehensive and ongoing management.
# MAGIC
# MAGIC - Hypertension is more prevalent in elective care, suggesting patients regularly manage this condition through planned medical visits.
# MAGIC - Diabetes is notable in urgent care, pointing to acute complications requiring prompt attention.
# MAGIC - Obesity is higher in emergency care, reflecting its potential to lead to severe, immediate health crises.

# COMMAND ----------

#Creating a graph of Medical Condition by Blood Type
from pyspark.sql.functions import count

# Grouping by Blood Type and Medical Condition
blood_type_condition_histogram = df.groupBy("Blood Type", "Medical Condition").agg(count("*").alias("Frequency"))

display(blood_type_condition_histogram)

# COMMAND ----------

# MAGIC %md
# MAGIC **Analysis:**
# MAGIC
# MAGIC - Diabetes is most frequent in A+ (1213), suggesting this blood type might have a higher prevalence for this condition.
# MAGIC - Asthma has the highest frequency in AB+ (1189), indicating a greater impact on this blood type.
# MAGIC - Arthritis is most common in B+ (1201), showing a significant occurrence in this group.
# MAGIC - Hypertension is highest in AB+ (1215), highlighting a particular vulnerability for this blood type.
# MAGIC - Obesity is more frequent in B- (1188), suggesting a notable correlation.
# MAGIC - Cancer shows a high occurrence in AB- (1198), indicating a significant presence within this blood type.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC **Exploratory SQL Queries:**
# MAGIC
# MAGIC SQL queries are used to identify patterns, such as the most common medical conditions by hospital and the medical conditions associated with the highest average billing amounts.
# MAGIC
# MAGIC **Visualization and Advanced Analysis:**
# MAGIC
# MAGIC Visualizations are created to graphically represent the distribution of billing amounts and the frequency of medical conditions across different hospitals.
# MAGIC
# MAGIC Complex SQL queries are executed to determine the costliest medical conditions by hospital and to understand the impact of age on billing amounts.
# MAGIC
# MAGIC

# COMMAND ----------

#Finding frequency of admissions of all hospitals
hospital_histogram = df.groupBy("Hospital").agg(count("*").alias("Frequency"))

display(hospital_histogram)

# COMMAND ----------

#Filtering Top 10 Hospital based on number of admissions
hospital_histogram.createOrReplaceTempView("hospital_histogram")
display(spark.sql("select * from hospital_histogram order by Frequency desc limit 10"))

# COMMAND ----------

# MAGIC %md
# MAGIC **Analysis:**
# MAGIC
# MAGIC - LLC Smith dominates the top 10 hospitals by admissions, reflecting a significant market presence.
# MAGIC
# MAGIC - Admissions are relatively balanced across the top 10, with a narrow range between 32 and 44.

# COMMAND ----------

#Creating a temporary view for running SQL queries
df.createOrReplaceTempView("medical_data")

# COMMAND ----------

#Using Spark SQL to find the top 10 hospitals with the highest average billing by medical coniditon
display(spark.sql("SELECT `Medical Condition`, Hospital, AVG(`Billing Amount`) AS Avg_Billing FROM medical_data GROUP BY `Medical Condition`, Hospital ORDER BY Avg_Billing DESC LIMIT 5"))

# COMMAND ----------

# MAGIC %md
# MAGIC **Analysis:**
# MAGIC
# MAGIC
# MAGIC - Griffin Group has the highest average billing for Hypertension at 52,764.28.
# MAGIC - Hernandez-Morton ranks second with Cancer treatments averaging 52,373.03.
# MAGIC - Sons and Bailey comes in third with Hypertension treatments averaging 52,271.66.
# MAGIC - PLC Garner is fourth, with an average billing for Asthma at 52,181.84.
# MAGIC - Walker-Garcia is fifth, having an average billing for Arthritis at 52,170.04.
# MAGIC
# MAGIC These findings highlight that Hypertension and Cancer are among the most expensive conditions to treat across different hospitals.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * from
# MAGIC (select Age, `Medical Condition`, AVG(`Billing Amount`) as Avg_Billing
# MAGIC FROM medical_data
# MAGIC GROUP BY Age, `Medical Condition`
# MAGIC ) subquery
# MAGIC ORDER BY Avg_Billing DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC **Analysis:**
# MAGIC
# MAGIC **1. Average, Minimum, and Maximum Billing Amount by Age:**
# MAGIC
# MAGIC - Age 15 shows the highest maximum billing amount of 44,389.8, the highest minimum billing of 24,743.6, and the highest average billing of 32,660.
# MAGIC - Average billing generally decreases and stabilizes within the range of 24,000 to 26,000 across most ages.
# MAGIC - At age 86, there is a slight increase in the average billing amount to 29,073.1.
# MAGIC - The lowest average billing amount is observed at age 89, at 14,892.
# MAGIC
# MAGIC **2. Age-Billing Correlation by Medical Condition:**
# MAGIC
# MAGIC There is no apparent correlation between age and billing amount across different medical conditions, indicating that billing is not strongly influenced by age for specific conditions.

# COMMAND ----------

# MAGIC %md
# MAGIC **Conclusion:**
# MAGIC
# MAGIC The analysis reveals key insights into the healthcare dataset, including billing trends, the prevalence of medical conditions and how these factors vary across different hospitals. The visualizations and SQL queries provide a clear understanding of the data, enabling data-driven decision-making for healthcare management and policy formulation.
# MAGIC
