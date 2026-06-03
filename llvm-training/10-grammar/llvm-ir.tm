language llvm(go);

lang = "llvm"
package = "github.com/llir/ll"
eventBased = true
eventFields = true
eventAST = true

# Textmapper grammar for LLVM IR.
# Used here as a reference for exact syntax productions. Consult
# the upstream `github.com/llir/ll` project for the canonical version.
#
# Conventions:
#   'keyword'        literal token
#   Foo              non-terminal
#   Foo?, Fooopt     optional
#   Foo*             zero or more
#   Foo+             one or more
#   (A separator B)+ list of A separated by B, at least one
#   Foo=Bar          captures Bar under name Foo in the AST
#   %interface Foo;  declares Foo as a category; any production marked
#                    `-> Foo` belongs to it

# ### [ Lexical part ] #########################################################

:: lexer

_ascii_letter_upper = /[A-Z]/
_ascii_letter_lower = /[a-z]/
_ascii_letter       = /{_ascii_letter_upper}|{_ascii_letter_lower}/
_letter             = /{_ascii_letter}|[-$\._]/
_escape_letter      = /{_letter}|[\\]/
_decimal_digit      = /[0-9]/
_hex_digit          = /{_decimal_digit}|[A-Fa-f]/

comment    : /[;][^\r\n]*/        (space)
whitespace : /[\x00 \t\r\n]+/     (space)

# === [ Identifiers ] ==========================================================

_name        = /{_letter}({_letter}|{_decimal_digit})*/
_escape_name = /{_escape_letter}({_escape_letter}|{_decimal_digit})*/
_quoted_name = /{_quoted_string}/
_id          = /{_decimals}/

# Global identifiers
global_ident_tok : /{_global_name}|{_global_id}/
_global_name = /[@]({_name}|{_quoted_name})/
_global_id   = /[@]{_id}/

# Local identifiers
local_ident_tok : /{_local_name}|{_local_id}/
_local_name = /[%]({_name}|{_quoted_name})/
_local_id   = /[%]{_id}/

# Labels — [-a-zA-Z$._0-9]+:
label_ident_tok : /(({_letter}|{_decimal_digit})({_letter}|{_decimal_digit})*[:])|({_quoted_string}[:])/   (class)

# Attribute group identifiers
attr_group_id_tok : /[#]{_id}/

# Comdat identifiers
comdat_name_tok : /[$]({_name}|{_quoted_name})/

# Metadata identifiers
metadata_name_tok : /[!]{_escape_name}/   (class)
metadata_id_tok   : /[!]{_id}/

# DWARF / DI identifier classes
dwarf_tag_tok          : /DW_TAG_({_ascii_letter}|{_decimal_digit}|[_])*/
dwarf_att_encoding_tok : /DW_ATE_({_ascii_letter}|{_decimal_digit}|[_])*/
di_flag_tok            : /DIFlag({_ascii_letter}|{_decimal_digit}|[_])*/
disp_flag_tok          : /DISPFlag({_ascii_letter}|{_decimal_digit}|[_])*/
dwarf_lang_tok         : /DW_LANG_({_ascii_letter}|{_decimal_digit}|[_])*/
dwarf_cc_tok           : /DW_CC_({_ascii_letter}|{_decimal_digit}|[_])*/
checksum_kind_tok      : /CSK_({_ascii_letter}|{_decimal_digit}|[_])*/
dwarf_virtuality_tok   : /DW_VIRTUALITY_({_ascii_letter}|{_decimal_digit}|[_])*/
dwarf_macinfo_tok      : /DW_MACINFO_({_ascii_letter}|{_decimal_digit}|[_])*/
dwarf_op_tok           : /DW_OP_({_ascii_letter}|{_decimal_digit}|[_])*/

emission_kind_tok   : /(DebugDirectivesOnly)|(FullDebug)|(LineTablesOnly)|(NoDebug)/
name_table_kind_tok : /(GNU)|(None)|(Default)/

# === [ Integer literals ] =====================================================
int_lit_tok    : /[-]?[0-9]+|{_int_hex_lit}/
_int_hex_lit   = /[us]0x{_hex_digit}+/
_decimals      = /{_decimal_digit}+/

# === [ Floating-point literals ] ==============================================
float_lit_tok  : /{_frac_lit}|{_sci_lit}|{_float_hex_lit}/
_frac_lit      = /{_sign}?{_decimals}[\.]{_decimal_digit}*/
_sign          = /[+-]/
_sci_lit       = /{_frac_lit}[eE]{_sign}?{_decimals}/
_float_hex_lit = /0x[KLMHR]?{_hex_digit}+/

# === [ String literals ] ======================================================
string_lit_tok  : /{_quoted_string}/
_quoted_string  = /["][^"]*["]/

# === [ Types ] ================================================================
int_type_tok : /i[0-9]+/

# --- [ Keywords ] (alphabetical, abridged for readability)
# 'nounwind' is listed first as a work-around for a Textmapper resolution order.
'nounwind' : /nounwind/

# (Hundreds of keyword token declarations elided here for brevity. The full
# upstream grammar at github.com/llir/ll declares one token per LLVM IR
# keyword: aarch64_sve_vector_pcs, acq_rel, acquire, add, addrspace, ...,
# zeroext, zeroinitializer, zext.)

# Punctuation
',' : /[,]/
'!' : /[!]/
'...' : /\.\.\./
'(' : /[(]/
')' : /[)]/
'[' : /[\[]/
']' : /[\]]/
'{' : /[{]/
'}' : /[}]/
'*' : /[*]/
'<' : /[<]/
'=' : /[=]/
'>' : /[>]/
pipe_tok : /[|]/

# Specialized metadata node names (DI*) and field-name tokens (align:, file:, line:, etc.)
# are declared inline in the upstream grammar. See github.com/llir/ll.

# ### [ Syntax part ] ##########################################################

:: parser
%input Module;

# === [ Identifiers ] ==========================================================
GlobalIdent   -> GlobalIdent   : global_ident_tok ;
LocalIdent    -> LocalIdent    : local_ident_tok ;
LabelIdent    -> LabelIdent    : label_ident_tok ;
AttrGroupID   -> AttrGroupID   : attr_group_id_tok ;
ComdatName    -> ComdatName    : comdat_name_tok ;
MetadataName  -> MetadataName  : metadata_name_tok ;
MetadataID    -> MetadataID    : metadata_id_tok ;

# === [ Literals ] =============================================================
BoolLit   -> BoolLit   : 'true' | 'false' ;
IntLit    -> IntLit    : int_lit_tok ;
UintLit   -> UintLit   : int_lit_tok ;
FloatLit  -> FloatLit  : float_lit_tok ;
StringLit -> StringLit : string_lit_tok ;
NullLit   -> NullLit   : 'null' ;

# === [ Module ] ===============================================================
# A module is a sequence of target definitions followed by top-level entities.
Module -> Module
  : TargetDefs=TargetDef* TopLevelEntities=TopLevelEntity*
;

%interface TopLevelEntity;
TopLevelEntity -> TopLevelEntity
  : ModuleAsm
  | TypeDef
  | ComdatDef
  | GlobalDecl
  | IndirectSymbolDef
  | FuncDecl
  | FuncDef
  | AttrGroupDef
  | NamedMetadataDef
  | MetadataDef
  | UseListOrder
  | UseListOrderBB
;

# --- Target definition ---
%interface TargetDef;
TargetDef -> TargetDef
  : SourceFilename | TargetDataLayout | TargetTriple
;
SourceFilename   -> SourceFilename   : 'source_filename' '=' Name=StringLit ;
TargetDataLayout -> TargetDataLayout : 'target' 'datalayout' '=' DataLayout=StringLit ;
TargetTriple     -> TargetTriple     : 'target' 'triple' '=' TargetTriple=StringLit ;

# Function definition (sketch — full version in upstream)
FuncDef -> FuncDef
  : 'define' Header=FuncHeader Metadata=MetadataAttachment* Body=FuncBody
;

FuncHeader -> FuncHeader
  : (Linkage | ExternLinkage)? Preemptionopt Visibilityopt DLLStorageClassopt
    CallingConvopt ReturnAttrs=ReturnAttribute*
    RetType=Type Name=GlobalIdent '(' Params ')'
    UnnamedAddropt AddrSpaceopt FuncHdrFields=FuncHdrField*
;

FuncBody -> FuncBody
  : '{' Blocks=BasicBlock+ UseListOrders=UseListOrder* '}'
;

BasicBlock -> BasicBlock
  : Name=LabelIdentopt Insts=Instruction* Term=Terminator
;

# === [ Types ] ================================================================
%interface Type;
Type -> Type : VoidType | FuncType | FirstClassType ;

%interface FirstClassType;
FirstClassType -> FirstClassType : ConcreteType | MetadataType ;

%interface ConcreteType;
ConcreteType -> ConcreteType
  : IntType | FloatType | PointerType | VectorType | LabelType
  | ArrayType | StructType | NamedType | MMXType | TokenType
;

VoidType     -> VoidType     : 'void' ;
IntType      -> IntType      : int_type_tok ;
FloatType    -> FloatType    : FloatKind ;
FloatKind    -> FloatKind    : 'half' | 'bfloat' | 'float' | 'double' | 'x86_fp80' | 'fp128' | 'ppc_fp128' ;
MMXType      -> MMXType      : 'x86_mmx' ;
LabelType    -> LabelType    : 'label' ;
TokenType    -> TokenType    : 'token' ;
MetadataType -> MetadataType : 'metadata' ;
NamedType    -> NamedType    : Name=LocalIdent ;
OpaqueType   -> OpaqueType   : 'opaque' ;

PointerType -> PointerType
  : Elem=Type AddrSpaceopt '*'
;

VectorType -> VectorType
  : '<' Len=UintLit 'x' Elem=Type '>'
  | '<' 'vscale' 'x' Len=UintLit 'x' Elem=Type '>'   -> ScalableVectorType
;

ArrayType -> ArrayType
  : '[' Len=UintLit 'x' Elem=Type ']'
;

StructType -> StructType
  : '{' Fields=(Type separator ',')+? '}'
  | '<' '{' Fields=(Type separator ',')+? '}' '>'   -> PackedStructType
;

FuncType -> FuncType
  : RetType=Type '(' Params ')'
;

# ... NOTE: this file is excerpted for readability. The upstream grammar
# at github.com/llir/ll covers ~1500 lines with full productions for:
#
#   * Every instruction (FNegInst, AddInst, FAddInst, ..., LandingPadInst,
#     CatchPadInst, CleanupPadInst)
#   * Every terminator (RetTerm, BrTerm, CondBrTerm, SwitchTerm,
#     IndirectBrTerm, InvokeTerm, CallBrTerm, ResumeTerm,
#     CatchSwitchTerm, CatchRetTerm, CleanupRetTerm, UnreachableTerm)
#   * Every metadata field for every DI* node
#   * Every attribute (FuncAttribute, ParamAttribute, ReturnAttribute)
#   * Every constant expression form
#
# Refer to upstream for the canonical grammar:
#   https://github.com/llir/ll
#
# This excerpted version captures the structural skeleton useful for
# answering "what's the shape of construct X?" without the full bulk.
