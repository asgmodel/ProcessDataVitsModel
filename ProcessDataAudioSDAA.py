import gradio as gr
import pandas as pd

from df.enhance import enhance, init_df, load_audio, save_audio
import librosa
import soundfile as sf
model_enhance, df_state, _ = init_df()
def  remove_nn(wav,sample_rate=16000):
    sf.write("full_generation.wav", wav,sample_rate,format='WAV')
    audio, _ = load_audio('full_generation.wav', sr=df_state.sr())
    enhanced = enhance(model_enhance, df_state, audio)
    save_audio("enhanced.wav", enhanced, df_state.sr())
    audiodata, samplerate = librosa.load("enhanced.wav", sr=16000)
    return 16000,audiodata

from gradio_client import Client, handle_file
def  get_text_from_audio(audio):
    
    sf.write("temp.wav", audio, 16000,format='WAV')
    
    client = Client("MohamedRashad/Arabic-Whisper-CodeSwitching-Edition")
    result = client.predict(
        inputs=handle_file('temp.wav'),
        api_name="/predict_1"
    )
    return result

class DataViewerApp:
    def __init__(self, df):  # Initialize with dataframes
        self.df=df
        self.data =df[['text','speaker_id']]
        self.sdata = df['audio'].to_list()  # Separate audio data storage
        self.current_page = 0
        self.current_selected = -1

    def finsh_data(self):
        self.df['audio'] = self.sdata
        self.df[['text','speaker_id']]=self.data

        return self.df

    def get_page_data(self, page_number):
        start_index = page_number * 10
        end_index = start_index + 10
        return self.data.iloc[start_index:end_index]

    def update_page(self, new_page):
        self.current_page = new_page
        return (
            self.get_page_data(self.current_page),
            self.current_page > 0,
            self.current_page < len(self.data) // 10 - 1,
            self.current_page
        )

    def trim_audio(self, text,data_oudio):
  # Load audio
        #row = self.data.iloc[self.current_selected]
        #row['text'] = text  # Use .loc for safer row modification
        #self.data.iloc[self.current_selected] = row
        sr,audio=data_oudio
        if sr!=16000:
            sf.write("tempppp.wav", audio, sr,format='WAV')
            #audio, samplerate = librosa.load("temp.wav", sr=16000)

        return "Trimmed audio saved to 'output.wav'"

    def on_select(self, evt: gr.SelectData):
        index_now = evt.index[0]
        self.current_selected = (self.current_page * 10) + index_now
        row = self.data.iloc[self.current_selected]
        row_audio = self.sdata[self.current_selected]
        return (16000, row_audio), row['text']

    def save_row(self, text,data_oudio):
        row = self.data.iloc[self.current_selected]
        row['text'] = text  # Use .loc for safer row modification
        self.data.iloc[self.current_selected] = row
        sr,audio=data_oudio
        if sr!=16000:
            sf.write("temp.wav", audio, sr,format='WAV')
            audio, samplerate = librosa.load("temp.wav", sr=16000)

        self.sdata[self.current_selected] = audio  # Update audio too

        return self.get_page_data(self.current_page)
    #from datasets import Dataset,DatasetDict
    def Convert_DataFreme_To_DataSet(self,namedata):
           df=self.finsh_data()
           if "_index_level_0_" in df.columns: 
                df =df.drop(columns=["_index_level_0_"])
           train_df =df

           eval_df =df.sample(1,random_state=42)
#eval_df = eval_df[train_df.columns]
#train_df = df.drop(eval_df.index)
#eval_df_single_row = pd.DataFrame(df.iloc[55]).transpose()
           ds = {
                "train": Dataset.from_pandas(train_df),
                 "eval": Dataset.from_pandas(eval_df),
                 "full_generation":Dataset.from_pandas(eval_df)
                 }

           dataset = DatasetDict(ds)
           dirr='/content/drive/MyDrive/vitsM/DATA/sata/'+namedata
           dataset.save_to_disk(dirr)
           return dirr
    def All_enhance(self):
        for i in range(0,len(self.sdata)):
              _,y=remove_nn(self.sdata[i])
              self.sdata[i]=y
        return self.data
    def delete_row(self):

        self.data.drop(self.current_selected, inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        self.df.drop(self.current_selected, inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self.sdata.pop(self.current_selected)
        self.current_selected = -1
        # self.audio_player.update(None)  # Clear audio player
        # self.txt_audio.update("")  # Clear text input

        return self.get_page_data(self.current_page)



    def get_output_audio(self):
        return self.sdata[self.current_selected] if self.current_selected >= 0 else None

    def create_interface(self):

        with gr.Blocks() as interface:
               # with gr.Column():
                   # with gr.Row():
                       # with gr.Column(scale=3):
                           # gr.Markdown("## Data Viewer")

                          # Data Table
                           # self.data_tablee = gr.List(  # Notice 'self.' here
                               # value=[[3],[4],[5],[6],[8]],
                             #   headers=['speker _id'],
                            # interactive=True
                          #  )
            with gr.Column():
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("## Data Viewer")

                        # Data Table
                        self.data_table = gr.DataFrame(  # Notice 'self.' here
                            value=self.get_page_data(self.current_page),
                            headers=list(self.data.columns),
                           # interactive=True
                        )


                # Audio Player
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("## Row Data")

                            self.txt_audio = gr.Textbox(label="Text", interactive=True)
                            self.audio_player = gr.Audio(label="Audio")
                            self.btn_del = gr.Button("delete", size="sm")
                            self.btn_save = gr.Button("save", size="sm")
                            self.btn_denoise = gr.Button("denoise", size="sm")
                            self.btn_enhance = gr.Button("enhance", size="sm")
                            self.btn_newsave=gr.Button("New cut save",size="sm")
                            self.btn_all_enhance=gr.Button("All enhance",size="sm" )
                            self.totext=gr.Button("to text",size="sm" )

                with gr.Row(equal_height=False):
                            self.prev_button = gr.Button("Previous Page",scale=1, size="sm")

                            self.page_number = gr.Number(value=self.current_page + 1, label="Page",scale=1)
                            self.next_button = gr.Button("Next Page",scale=1, size="sm")
                with gr.Row(equal_height=False):
                     with gr.Column(scale=1):


                            self.txt_dataset = gr.Textbox(label="Name DataSet", interactive=True)
                            self.btn_convertDataset= gr.Button("Save DataSet", size="sm")
                            self.label_dataset=gr.Label()
            # ... (your existing Gradio layout code remains largely the same, but use 'self.' to access components)

        # Event handlers (use 'self.' to call methods of the class)
            self.data_table.select(self.on_select, None, [self.audio_player, self.txt_audio])
            self.prev_button.click(lambda page: self.update_page(page - 1), [self.page_number], [self.data_table, self.prev_button, self.next_button, self.page_number])
            self.next_button.click(lambda page: self.update_page(page + 1), [self.page_number], [self.data_table, self.prev_button, self.next_button, self.page_number])
            self.btn_save.click(self.save_row, [self.txt_audio,self.audio_player], [self.data_table])
            self.btn_all_enhance.click(self.All_enhance,[],[self.data_table])
            self.btn_newsave.click(self.trim_audio,[self.txt_audio,self.audio_player],[self.txt_audio])
            self.btn_del.click(self.delete_row, [], self.data_table)
            self.btn_denoise.click(lambda: remove_nn(self.get_output_audio()), [], self.audio_player)
            self.btn_convertDataset.click(self.Convert_DataFreme_To_DataSet,[self.txt_dataset],[self.label_dataset])
            self.totext.click(lambda: get_text_from_audio(self.get_output_audio()), [], self.txt_audio)
             # Update table after deletion
        # ... (rest of your event handlers)

        return interface
