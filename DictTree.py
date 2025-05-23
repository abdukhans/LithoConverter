import string

alphanums = string.ascii_letters + string.digits

ALPHABETS =alphanums


class LetterNode:

    def __init__(self, letter:str,is_word:bool,freq=0) -> None:

        self.letter = letter
        self.is_word = is_word
        self.word_freq = freq

        self.depth = 0 

        self.subset_pooled_freq = freq

       
        self.childern:dict[str,LetterNode| None] = { c:None for c in ALPHABETS   }

        self.key_word_path      = letter

        self.similar_words = []

        self.parent = None

        self.is_compressed = False

        self.idx = 0 

    def addWord(self,word:str,freq=0):


        cur_node = self
        for idx,char in enumerate(word):
            if (cur_node.childern[char]):
               
                cur_node = cur_node.childern[char]
        
            else:

                cur_node.childern[char] = LetterNode(char,False,freq=freq)
                cur_node = cur_node.childern[char]
            

            cur_node.key_word_path =word[:idx+1]
        cur_node.similar_words.append(word)  



        cur_node.is_word = True

        # self.update_freq()
        

    def update_freq(self):

        for char in self.childern:
            child = self.childern[char]
            if child:
                child.update_freq()
                self.subset_pooled_freq += child.subset_pooled_freq
        self.subset_pooled_freq += self.word_freq
            



    def get_words(self) -> list[str]:

        res = []
        
        for char in ALPHABETS:

            child = self.childern[char]


            if child:
                list_of_words = child.get_words()
                res.extend( f"{self.letter}{sub_word}" for sub_word in list_of_words   )


        if self.is_word:
            res.append(self.letter)
                    
        return res

    def get_node(self,word:str) -> "LetterNode": 
        cur_node = self   
        for idx,char in enumerate(word):
                if cur_node == None:
                    raise Exception(f"The child for '{word[idx-1]}' is not defined for that path '{word[:idx-1]}'  ")

        
                cur_node = cur_node.childern[char]

        if cur_node == None:
             raise Exception(f"The child for '{char}' is not defined for that path '{word[:idx]}'  ")

        return cur_node

    def get_all_key_words_with_start(self,word):

        node_with_start = self.get_node(word)


        return [f"{word[:-1]}{i}"for i in node_with_start.get_words()]

    def Compress_Node(self):

        if self.is_compressed:
            return

        for child_char in self.childern:
            child = self.childern[child_char]
            if child:
                child.Compress_Node()
                self.similar_words.extend(child.similar_words)

                # self.word_freq += child.word_freq

                
        self.is_compressed = True
    
        pass

    def get_tot_nodes(self) -> int:

        total_nodes = 1


        for child_char in self.childern:

            child = self.childern[child_char]

            if child:
                total_nodes += child.get_tot_nodes()

        return total_nodes

    def get_total_words(self) -> int:

        return len(self.get_all_key_words_with_start(''))

                


if __name__ == '__main__':

    tree = LetterNode('',False)

    tree.addWord('words')
    tree.addWord('wor')
    tree.addWord('ads')
    tree.addWord('adsd')
    print(tree.get_words())
    
    print(tree.childern['w'].get_words())
    
    print(tree.childern['a'].get_words())

    # print(tree.childern['a'].childern['d'].get_words())
    print("asdf", tree.get_all_key_words_with_start('ads'))
    print('key_words',tree.get_node('wor').key_word_path)
    tree.get_node('w').Compress_Node()
    print('sim_word',tree.get_node('w').similar_words)
    print('tot_nodes',tree.get_tot_nodes())
    print('tot_words',tree.get_node('adsd').get_total_words())

    # print(tree.get_node('ada').get_words())
    
        