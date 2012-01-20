#
# Contain a class that provides a set of common library functions for AST processing
#

import sys
import orio.module.loop.ast, orio.module.loop.oost
from orio.main.util.globals import *

#-----------------------------------------------------------
 
class CommonLib:
    '''A common library set for AST processing'''
    
    def __init__(self):
        '''To instantiate a common library object'''
        pass

    #-------------------------------------------------------

    def replaceIdent(self, tnode, iname_from, iname_to):
        '''Replace the names of all matching identifiers with the given name'''

        if isinstance(tnode, orio.module.loop.ast.NumLitExp):
            return tnode
            
        elif isinstance(tnode, orio.module.loop.ast.StringLitExp):
            return tnode
            
        elif isinstance(tnode, orio.module.loop.ast.IdentExp):
            if tnode.name == iname_from:
                tnode.name = iname_to
            return tnode
            
        elif isinstance(tnode, orio.module.loop.ast.ArrayRefExp):
            tnode.exp = self.replaceIdent(tnode.exp, iname_from, iname_to)
            tnode.sub_exp = self.replaceIdent(tnode.sub_exp, iname_from, iname_to)
            return tnode
            
        elif isinstance(tnode, orio.module.loop.ast.FunCallExp):
            tnode.exp = self.replaceIdent(tnode.exp, iname_from, iname_to)
            tnode.args = [self.replaceIdent(a, iname_from, iname_to) for a in tnode.args]
            return tnode
            
        elif isinstance(tnode, orio.module.loop.ast.UnaryExp):
            tnode.exp = self.replaceIdent(tnode.exp, iname_from, iname_to)
            return tnode
            
        elif isinstance(tnode, orio.module.loop.ast.BinOpExp):
            tnode.lhs = self.replaceIdent(tnode.lhs, iname_from, iname_to)
            tnode.rhs = self.replaceIdent(tnode.rhs, iname_from, iname_to)
            return tnode
            
        elif isinstance(tnode, orio.module.loop.ast.ParenthExp):
            tnode.exp = self.replaceIdent(tnode.exp, iname_from, iname_to)
            return tnode
            
        elif isinstance(tnode, orio.module.loop.ast.ExpStmt):
            if tnode.exp:
                tnode.exp = self.replaceIdent(tnode.exp, iname_from, iname_to)
            return tnode
            
        elif isinstance(tnode, orio.module.loop.ast.CompStmt):
            tnode.stmts = [self.replaceIdent(s, iname_from, iname_to) for s in tnode.stmts]
            return tnode
            
        elif isinstance(tnode, orio.module.loop.ast.IfStmt):
            tnode.test = self.replaceIdent(tnode.test, iname_from, iname_to)
            tnode.true_stmt = self.replaceIdent(tnode.true_stmt, iname_from, iname_to)
            if tnode.false_stmt:
                tnode.false_stmt = self.replaceIdent(tnode.false_stmt, iname_from, iname_to)
            return tnode
            
        elif isinstance(tnode, orio.module.loop.ast.ForStmt):
            if tnode.init:
                tnode.init = self.replaceIdent(tnode.init, iname_from, iname_to)
            if tnode.test:
                tnode.test = self.replaceIdent(tnode.test, iname_from, iname_to)
            if tnode.iter:
                tnode.iter = self.replaceIdent(tnode.iter, iname_from, iname_to)
            tnode.stmt = self.replaceIdent(tnode.stmt, iname_from, iname_to)
            return tnode

        elif isinstance(tnode, orio.module.loop.ast.TransformStmt):
            err('orio.module.loop.ast_lib.common_lib internal error:  unexpected AST type: "%s"' % tnode.__class__.__name__)
        
        elif isinstance(tnode, orio.module.loop.ast.NewAST):
            return tnode

        elif isinstance(tnode, orio.module.loop.ast.Comment):
            return tnode

        else:
            err('orio.module.loop.ast_lib.common_lib internal error:  unexpected AST type: "%s"' % tnode.__class__.__name__)
        
    #-------------------------------------------------------

    def containIdentName(self, exp, iname):
        '''
        Check if the given expression contains an identifier whose name matches to the given name
        '''

        if exp == None:
            return False
        
        if isinstance(exp, orio.module.loop.ast.NumLitExp):
            return False
        
        elif isinstance(exp, orio.module.loop.ast.StringLitExp):
            return False
        
        elif isinstance(exp, orio.module.loop.ast.IdentExp):
            return exp.name == iname
        
        elif isinstance(exp, orio.module.loop.ast.ArrayRefExp):
            return self.containIdentName(exp.exp, iname) or self.containIdentName(exp.sub_exp, iname)
        
        elif isinstance(exp, orio.module.loop.ast.FunCallExp):
            has_match = reduce(lambda x,y: x or y,
                               [self.containIdentName(a, iname) for a in exp.args],
                               False)
            return self.containIdentName(exp.exp, iname) or has_match
        
        elif isinstance(exp, orio.module.loop.ast.UnaryExp):
            return self.containIdentName(exp.exp, iname)
        
        elif isinstance(exp, orio.module.loop.ast.BinOpExp):
            return self.containIdentName(exp.lhs, iname) or self.containIdentName(exp.rhs, iname)
        
        elif isinstance(exp, orio.module.loop.ast.ParenthExp):
            return self.containIdentName(exp.exp, iname)
        
        elif isinstance(exp, orio.module.loop.ast.NewAST):
            return False
        
        elif isinstance(exp, orio.module.loop.ast.Comment):
            return False

        else:
            err('orio.module.loop.ast_lib.common_lib internal error:  unexpected AST type: "%s"' % exp.__class__.__name__)
            
    #-------------------------------------------------------

    def isComplexExp(self, exp):
        '''
        To determine if the given expression is complex. Simple expressions contain only a variable
        or a number or a string.
        '''
        
        if isinstance(exp, orio.module.loop.ast.NumLitExp):
            return False
        
        # a rare case
        elif isinstance(exp, orio.module.loop.ast.StringLitExp):
            return False
        
        elif isinstance(exp, orio.module.loop.ast.IdentExp):
            return False
        
        # a rare case
        elif isinstance(exp, orio.module.loop.ast.ArrayRefExp):
            return True
        
        elif isinstance(exp, orio.module.loop.ast.FunCallExp):
            return True
        
        elif isinstance(exp, orio.module.loop.ast.UnaryExp):
            return self.isComplexExp(exp.exp)
        
        elif isinstance(exp, orio.module.loop.ast.BinOpExp):
            return True
        
        elif isinstance(exp, orio.module.loop.ast.ParenthExp):
            return self.isComplexExp(exp.exp)
        
        # a rare case
        elif isinstance(exp, orio.module.loop.ast.NewAST):
            return True
        
        elif isinstance(exp, orio.module.loop.ast.Comment):
            return False

        else:
            err('orio.module.loop.ast_lib.common_lib internal error:  unexpected AST type: "%s"' % exp.__class__.__name__)
            
    #-------------------------------------------------------

    def collectIdents(self, exp):
        '''
        To collect all identifiers within the given expression.
        '''
        
        if isinstance(exp, orio.module.loop.ast.NumLitExp):
            return []
        
        elif isinstance(exp, orio.module.loop.ast.StringLitExp):
            return []
        
        elif isinstance(exp, orio.module.loop.ast.IdentExp):
            return [exp.name]
        
        elif isinstance(exp, orio.module.loop.ast.ArrayRefExp):
            return self.collectIdents(exp.exp) + self.collectIdents(exp.sub_exp)

        elif isinstance(exp, orio.module.loop.ast.FunCallExp):
            ids = reduce(lambda x,y: x + y,
                         [self.collectIdents(a) for a in exp.args],
                         [])
            return ids
        
        elif isinstance(exp, orio.module.loop.ast.UnaryExp):
            return self.collectIdents(exp.exp)
        
        elif isinstance(exp, orio.module.loop.ast.BinOpExp):
            return self.collectIdents(exp.lhs) + self.collectIdents(exp.rhs)
        
        elif isinstance(exp, orio.module.loop.ast.ParenthExp):
            return self.collectIdents(exp.exp)
        
        else:
            err('orio.module.loop.ast_lib.common_lib.collectIdents: unexpected AST type: "%s"' % exp.__class__.__name__)
            
            
    #-------------------------------------------------------


class NodeMapper(orio.module.loop.oost.NodeVisitor):
    """ A node visitor that applies a given function to every node in the tree
    """
    def __init__(self, fun):
        self.fun = fun

    def visit(self, node):
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)
        
    def generic_visit(self, node):
        self.fun(node)
        for c in node.kids:
            self.visit(c)

