from DecisionTreeNode import DecisionTreeNode

def render_as_dot(node: DecisionTreeNode, file_path: str) -> None:
    '''Writes the decision tree out in GraphViz-dot format'''
    with open(file_path) as file:
        global dot_node_counter
        dot_node_counter = 0
        dotify_helper(node, file)
    #:with
#:dotify()

def dotify_helper(node: DecisionTreeNode,
                  file,
                  counter: int =0,
                  parent_node_name: str = None):
    node_name = f'node_{dot_node_counter}'
    print(file=file) # blank line
    print(f'{node_name}[label="{node.label()}"]', file=file)
    counter += 1
    if parent_node_name:
        print(f'{parent_node_name} -> {node_name}')
    #:if
    if node.is_split():
        assert len(node.children) == 2
        for child in node.children:
            counter = dotify_helper(child, file, counter, node_name)
    #:for
    return counter
#:dotify_helper()

