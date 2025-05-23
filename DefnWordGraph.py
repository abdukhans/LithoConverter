
from graphviz import Digraph
dot = Digraph(engine='neato')

dot.attr(overlap="scale")
class WordNode:

    def __init__(self, word:str, is_def_word:bool = False ) -> None:
        self.word = word
        self.is_def_word = is_def_word

        self.def_words  = []
    
    
class DefnWordGraph:



    def __init__(self,show_cycles=False) -> None:



        self.show_cycles = show_cycles

        self.adj_array:dict[str,list[str]] = {}

        self.word_node_map:dict[str,WordNode] = {}

        


    def add_word(self,word:str) -> None:

        if word in self.adj_array:
            return
        

        self.adj_array[word]     = []
        self.word_node_map[word] = WordNode(word)

  
    def add_def_word(self,main_word:str,defn_word:str) -> None:
    
        self.adj_array[main_word].append(defn_word)


        self.add_word(defn_word)

    def add_def_words(self,main_word:str,defn_words:list[str]) -> None:


        new_defn_words = [i for i in defn_words if i not in  self.adj_array[main_word]]

        self.adj_array[main_word].extend(new_defn_words)

        for i in defn_words:

                self.add_word(i)

    def render_graph(self,out_svg='Word_graph'):

        for node in self.adj_array:

            dot.node(f"{node}", f"Node: {node}\nDef Words:{self.word_node_map[node].def_words}") 
        

        for node in  self.adj_array:
            for neighbour_node in  self.adj_array[node]:
                dot.edge(f"{node}",neighbour_node)

        
        
        dot.render(out_svg, format='svg')


    def get_def_words(self,main_word:str,regen_visited=False):


        # if self.visited[main_word]:
        #     return


        # if regen_visited:
        #     self.visited = { string:False  for string in self.adj_array}

        
        # if len(self.adj_array[main_word]) == 0:
        #     self.word_node_map[main_word].def_words.append(main_word)
        # else:
        #     for n_node in self.adj_array[main_word]:

        #         if not(self.visited[n_node]):
        #             self.get_def_words(n_node)


        #         self.word_node_map[main_word].def_words.extend(self.word_node_map[n_node].def_words)

        # self.visited[main_word] = True

          


        dfs_stack = [[main_word,0]]

        hash_stack = {main_word}

        while len(dfs_stack) > 0:


            cur_node,neigh_idx = dfs_stack[-1][0],dfs_stack[-1][1]

            

            if neigh_idx < len(self.adj_array[cur_node]):
                
                neigh_node = self.adj_array[cur_node][neigh_idx]

                if neigh_node in hash_stack:

                    if self.show_cycles:
                        print(f"'{cur_node=}' forms a cycle with '{neigh_node=}' so we are going to skip ")

                        cycle_path = []


                        for i in dfs_stack[::-1]:
                            node = i[0]
                            cycle_path.append(node)
                            if node == neigh_node:
                                break
                            


                        print(f"Cycle Path:\n{cycle_path[::-1]}\n")
                    dfs_stack[-1][1]+= 1
                    continue


                if self.visited[neigh_node]:
                    dfs_stack[-1][1]+= 1

                    new_defn_words = [i for i in self.word_node_map[neigh_node].def_words if i not in self.word_node_map[cur_node].def_words ]
                    self.word_node_map[cur_node].def_words.extend(new_defn_words)

                else:
                    dfs_stack.append([neigh_node,0])
                    hash_stack.add(neigh_node)


                
            else:
                if neigh_idx == 0:
                    self.word_node_map[cur_node].def_words.append(cur_node)
                    pass

                self.visited[cur_node] = True
                dfs_stack.pop()
                hash_stack.remove(cur_node)




            # for n_node in self.adj_array[cur_node]:
            #     if not(self.visited[n_node]):
            #         done = False
            #         dfs_stack.append(n_node)
            #         break
            #     pass
            # if done:
                
            #     is_root_word = len(self.adj_array[cur_node]) == 0 
                
            #     if is_root_word:
            #         self.word_node_map[cur_node].def_words.append(cur_node)
               
            #     else:
            #         for n_node in self.adj_array[cur_node]:

            #             self.word_node_map[cur_node].def_words.extend(self.word_node_map[n_node].def_words)

            #     self.visited[cur_node] = True
            #     dfs_stack.pop()
        pass



    def get_all_def_words(self):
        self.visited = { string:False  for string in self.adj_array}
        for word in self.adj_array:
            if not(self.visited[word]):
                self.get_def_words(word)




if __name__ == "__main__":


    GRAPH = DefnWordGraph()

    GRAPH.add_word('igneous')
    GRAPH.add_word('basalt')
    GRAPH.add_word('vein')
    

    def_words = [ i.lstrip().rstrip() for i in "molten, rock".split(',')]
    GRAPH.add_def_words('igneous',def_words)
    def_words = [ i.lstrip().rstrip() for i in "igneous, volcanic, rock".split(',')]
    GRAPH.add_def_words('basalt',def_words)

    GRAPH.get_all_def_words()
    GRAPH.render_graph()
    pass