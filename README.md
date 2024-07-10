# ... (your data loading and preprocessing code)
# Assuming you have dataframes named 'data' and 'sdata'
app = DataViewerApp(df)
interface = app.create_interface()
interface.launch()
