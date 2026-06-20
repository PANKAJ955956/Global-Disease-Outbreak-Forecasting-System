# GDOFS Power BI Dashboard Implementation Guide

This guide details the blueprint and technical steps required to create the GDOFS Executive Outbreak Dashboard inside **Power BI Desktop**, including data connections, custom DAX measures, visual layouts, and publishing schemas.

---

## 📂 Data Connection Setup
1. Open **Power BI Desktop**.
2. Click **Get Data** ➡️ **Text/CSV**.
3. Navigate to the project folder and select the featured dataset:
   `f:/NOIDA SEC 63/SMART BRIDGE PVT. LTD/PROJECTS/Global Disease Outbreak/data/processed/featured_data.csv`
4. Verify character encoding is set to `UTF-8` and click **Load**.

---

## 📐 DAX Calculations & Measures

To build the executive metrics, create the following calculated column and measures in the Power BI Model tab:

### 1. Case Fatality Rate (CFR) Measure
*   **Purpose:** Dynamically computes the percentage of cases resulting in deaths, adjusted by slicer filters.
*   **DAX Code:**
    ```dax
    CFR = DIVIDE(
        SUM(featured_data[deaths]), 
        SUM(featured_data[cases])
    ) * 100
    ```

### 2. Normalized Case Rate Average Measure
*   **Purpose:** Visualizes the cumulative scale of the infection adjusted per 100,000 population.
*   **DAX Code:**
    ```dax
    Avg_Case_Rate = AVERAGE(featured_data[case_rate])
    ```

### 3. Risk Label Category (Calculated Column)
*   **Purpose:** Maps numeric predictions back to risk names for matrix row aggregations.
*   **DAX Code:**
    ```dax
    Risk_Level_Name = SWITCH(
        featured_data[risk_label],
        0, "Low",
        1, "Medium",
        2, "High",
        3, "Critical",
        "Unknown"
    )
    ```

---

## 📊 Dashboard Visual Layout Blueprint

Arrange the canvas into a standard grid dashboard:

### 1. Executive Summary Cards (Top Row)
*   **Card 1 (Total Cases):** Field = `SUM(cases)`. Format as integer with thousands separator.
*   **Card 2 (Total Deaths):** Field = `SUM(deaths)`. Format as integer.
*   **Card 3 (Global CFR):** Field = `[CFR]` measure. Format as percentage suffix (`%`).
*   **Card 4 (Avg Risk Score):** Field = `AVERAGE(risk_score)`. Format to 1 decimal place.

### 2. Interactive Slicers (Left Sidebar or Top Banner)
*   **Disease Slicer:** Field = `disease` (Dropdown or list tiles).
*   **Year Slicer:** Field = `year` (Slider or list tiles).
*   **Country Slicer:** Field = `country` (Multi-select dropdown).

### 3. Geo-Spatial Bubble Map (Center Canvas)
*   **Visualization:** Bubble Map / Azure Map.
*   **Location Fields:**
    *   Latitude = `COUNTRY_COORDS` mapping or simple location mapping on `country`.
    *   Bubble Size = `[Avg_Case_Rate]` or `SUM(cases)`.
    *   Bubble Color Saturation = `[CFR]`.
*   **Tooltip:** Add `country`, `cases`, `deaths`, and `[CFR]`.

### 4. Disease Trend Timeline (Bottom Canvas)
*   **Visualization:** Area Chart or Line Chart.
*   **X-Axis:** `date` (formatted as Timeline hierarchy: Year > Quarter > Month).
*   **Y-Axis:** `SUM(cases)`.
*   **Legend (Split by):** `disease`.

### 5. Outbreak Risk Matrix Table (Right Sidebar)
*   **Visualization:** Matrix Table.
*   **Rows:** `country` ➡️ `disease`.
*   **Columns:** `[Risk_Level_Name]`.
*   **Values:** `SUM(cases)`.
*   **Conditional Formatting:** Apply a red-gradient background to "Critical" and "High" column values for fast scanning.

---

## 🚀 Publishing to Power BI Service
1. Click the **Publish** button on the Home tab ribbon.
2. Log in with your corporate or organizational Power BI account.
3. Select **My Workspace** or a shared department workspace.
4. Open the generated report link to set up scheduled daily refreshes connected to the SQLite database via a Power BI Gateway.
