# Delaware Educator Data Analysis (DataQuest 2026)

This project analyzes educator salary and mobility data provided by the Delaware Department of Education. The dataset contains approximately a few million records and is used to explore trends in educator salary, turnover, and movement across districts.

---

## Data Sources

This project uses publicly available datasets from the Delaware Department of Education:

* Educator Salary Data
  https://data.delaware.gov/Education/Educator-Average-Salary/rv4m-vy79/about_data

* Educator Mobility Data
  https://data.delaware.gov/Education/Educator-Mobility/jdcc-w6wr/about_data

---

## Setup Instructions

### 1. Download the datasets

Download both CSV files from the links above and place them in the project directory.

Rename them exactly as follows:

* `Educator_Salary.csv`
* `Educator_Mobility.csv`

---

### 2. Install required dependencies

Install the necessary Python libraries:

```bash
pip install numpy pandas matplotlib seaborn
```

---

### 3. Run the data processing script

Execute the dataset merging script:

```bash
python datasetMerger.py
```

This will generate the merged dataset used for analysis in the other files.

---

## 📌 Project Purpose

The goal of this project is to explore relationships between educator salary, experience, and mobility patterns across Delaware school districts using real-world administrative data.

---

## 🛠 Tools Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn

---

* The raw datasets are large (~3 million rows total), so processing may take time depending on system performance.
* Ensure sufficient memory is available when running the scripts.
