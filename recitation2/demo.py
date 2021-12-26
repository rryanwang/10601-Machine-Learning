# This class represents an individual node

class Node:
    def __init__(self, key):
        self.left = None
        self.right = None
        self.val = key


# this is that the tree looks like:
#               1
#              / \
#             2   3
#            / \
#           4   5

# height of tree is 2 (max # of edges between root and any given node)
# depth is 1 at node(2), depth is 2 at node(4)


root = Node(1)
root.left = Node(2)
root.right = Node(3)
root.left.left = Node(4)
root.left.right = Node(5)

def printPreorder(root):
    if root is not None:
        # first print the data of the node
        print(root.val, '\t', end='')

        # then recur on the left child
        printPreorder(root.left)

        # then recur on the right child
        printPreorder(root.right)

def printInorder(root):
    if root is not None:
        # first recur on left child
        printInorder(root.left)

        # then print the data of node
        print(root.val, '\t', end='')

        # now recur on the right child
        printInorder(root.right)

def printPostorder(root):
    if root:
        # first recur on the left child
        printPostorder(root.left)

        # then recur on the right child
        printPostorder(root.right)

        # then print the data of the node
        print(root.val, '\t', end='')


input('press any key to display Preorder traversal')
print('Preorder traversal of binary tree is: ')
printPreorder(root)

print('\n')

input('press any key to display Inorder traversal')
print('Inorder traversal of binary tree is: ')
printInorder(root)

print('\n')

input('press any key to display Postorder traversal')
print('Postorder traversal of binary tree is: ')
printPostorder(root)
