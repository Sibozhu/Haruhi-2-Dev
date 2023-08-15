from ChromaDB import ChromaDB
from LangChainGPT import LangChainGPT
import os

from utils import luotuo_openai_embedding, tiktoken

class ChatHaruhi:

    def __init__(self, system_prompt, \
                 story_db=None, story_text_folder = None, \
                 llm = 'openai', \
                 max_len_story = None, max_len_history = None):

        self.system_prompt = system_prompt

        if story_db:
            self.db = ChromaDB() 
            self.db.load(story_db)
        elif story_text_folder:
            # print("Building story database from texts...")
            self.db = self.build_story_db(story_text_folder) 
        else:
            raise ValueError("Either story_db or story_text_folder must be provided")
        
        
        if llm == 'openai':
            # self.llm = LangChainGPT()
            self.llm, self.embedding, self.tokenizer = self.get_models('openai')
        elif llm == 'debug':
            from PrintLLM import PrintLLM
            self.llm = PrintLLM()
            _, self.embedding, self.tokenizer = self.get_models('openai')
        else:
            print(f'warning! undefined llm {llm}, use openai instead.')
            self.llm, self.embedding, self.tokenizer = self.get_models('openai')

        self.max_len_story, self.max_len_history = self.get_tokenlen_setting('openai')

        if max_len_history is not None:
            self.max_len_history = max_len_history
            # user setting will override default setting

        if max_len_story is not None:
            self.max_len_story = max_len_story
            # user setting will override default setting

        self.dialogue_history = []

        # constants
        self.story_prefix_prompt = "Classic scenes for the role are as follows:"
        self.k_search = 19
        self.narrator = ['旁白', '', 'scene','Scene','narrator' , 'Narrator']
        self.dialogue_divide_token = '\n###\n'
        self.dialogue_bra_token = '「'
        self.dialogue_ket_token = '」'

    def get_models(self, model_name):
        # return the combination of llm, embedding and tokenizer
        if model_name == 'openai':
            return (LangChainGPT(), luotuo_openai_embedding, tiktoken)
        else:
            print(f'warning! undefined model {model_name}, use openai instead.')
            return (LangChainGPT(), luotuo_openai_embedding, tiktoken)
        
    def get_tokenlen_setting( self, model_name ):
        # return the setting of story and history token length
        if model_name == 'openai':
            return (1500, 1200)
        else:
            print(f'warning! undefined model {model_name}, use openai instead.')
            return (1500, 1200)

    def build_story_db(self, text_folder):
        # 实现读取文本文件夹,抽取向量的逻辑
        db = ChromaDB()

        strs = []

        # scan all txt file from text_folder
        for file in os.listdir(text_folder):
            # if file name end with txt
            if file.endswith(".txt"):
                file_path = os.path.join(text_folder, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    strs.append(f.read())

        vecs = []

        for mystr in strs:
            vecs.append(self.embedding(mystr))

        db.init_from_docs(vecs, strs)

        return db
    
    def save_story_db(self, db_path):
        self.db.save(db_path)
        
    def chat(self, text, role):
        # add system prompt
        self.llm.initialize_message()
        self.llm.system_message(self.system_prompt)

        # add story
        query = self.get_query_string(text, role)
        self.add_story( query )

        # add history
        self.add_history()

        # get response
        response = self.llm.get_response()

        # record dialogue history
        self.dialogue_history.append((query, response))
        
        return response
    
    def get_query_string(self, text, role):
        if role in self.narrator:
            return ":" + text
        else:
            return f"{role}:{self.dialogue_bra_token}{text}{self.dialogue_ket_token}"
        
    def add_story(self, query):
        query_vec = self.embedding(query)

        stories = self.db.search(query_vec, self.k_search)
        
        story_string = self.story_prefix_prompt
        sum_story_token = self.tokenizer(story_string)
        
        for story in stories:
            story_token = self.tokenizer(story) + self.tokenizer(self.dialogue_divide_token)
            if sum_story_token + story_token > self.max_len_story:
                break
            else:
                sum_story_token += story_token
                story_string += story + self.dialogue_divide_token

        self.llm.user_message(story_string)
        
    def add_history(self):
        sum_history_token = 0
        flag = 0
        for (query, response) in self.dialogue_history.reverse():
            current_count = self.tokenizer(query.split()) + self.tokenizer(response.split())
            sum_history_token += current_count
            if sum_history_token > self.max_len_history:
                break
            else:
                flag += 1

        if flag == 0:
            print('warning! no history added. the last dialogue is too long.')

        for (query, response) in self.dialogue_history[-flag:]:
            self.llm.ai_message(query)
            self.llm.user_message(response)

        
        