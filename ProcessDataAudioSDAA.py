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

    def delete_row(self):
        
        self.data.drop(self.current_selected, inplace=True)
        self.data.reset_index(drop=True, inplace=True)

        self.sdata.pop(self.current_selected)
        self.current_selected = -1
        # self.audio_player.update(None)  # Clear audio player
        # self.txt_audio.update("")  # Clear text input

        return self.get_page_data(self.current_page)
        

   
    def get_output_audio(self):
        return self.sdata[self.current_selected] if self.current_selected >= 0 else None

    def create_interface(self):
       
        with gr.Blocks() as interface:
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
                with gr.Row(equal_height=False):
                            self.prev_button = gr.Button("Previous Page",scale=1, size="sm")  
                            
                            self.page_number = gr.Number(value=self.current_page + 1, label="Page",scale=1)
                            self.next_button = gr.Button("Next Page",scale=1, size="sm")
                      
            # ... (your existing Gradio layout code remains largely the same, but use 'self.' to access components)

        # Event handlers (use 'self.' to call methods of the class)
            self.data_table.select(self.on_select, None, [self.audio_player, self.txt_audio])
            self.prev_button.click(lambda page: self.update_page(page - 1), [self.page_number], [self.data_table, self.prev_button, self.next_button, self.page_number])
            self.next_button.click(lambda page: self.update_page(page + 1), [self.page_number], [self.data_table, self.prev_button, self.next_button, self.page_number])
            self.btn_save.click(self.save_row, [self.txt_audio,self.audio_player], [self.data_table])
            self.btn_del.click(self.delete_row, [], self.data_table) 
            self.btn_denoise.click(lambda: remove_nn(self.get_output_audio()), [], self.audio_player)
             # Update table after deletion
        # ... (rest of your event handlers)

        return interface
