from EDA.DataStructure import DataStructureAnalyzer
from Infrastructure.ConsolePrinter import ConsolePrinter
from EDA.statistics import DataStats
from EDA.data_preprocessing import load_data

if __name__ == "__main__":
    # Load data
    image_path = r'data/faces/'
    labels_path = r"data/faces/labels.csv"
    labels = load_data(labels_path)

    # Run EDA structure analysis
    eda = DataStructureAnalyzer(labels)
    output = ConsolePrinter(eda)

    # 1️⃣ Print everything automatically:
    #output.console()

    # 2️⃣ Or only print certain methods:
    #output.console(["overview", "column_missing_values"])

output.console(["dtypes"])