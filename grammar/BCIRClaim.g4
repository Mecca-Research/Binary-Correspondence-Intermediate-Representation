grammar BCIRClaim;

module      : 'MODULE' ID '{' header registry phaseBlock+ '}' EOF ;
header      : 'HEADER' '{' headerField+ '}' ;
headerField : ID '=' (INT | STRING) ';' ;
registry    : 'REGISTRY' 'RES' '{' record+ '}' ;
record      : 'RECORD' ID ':' 'RES' '{' recordField+ '}' ;
recordField : ID '=' value ';' ;
phaseBlock  : 'PHASE' INT '{' claim+ '}' ;
claim       : 'CLAIM' INT ':' ID '(' argList? ')' '::' '{' contractField+ '}' ;
argList     : arg (',' arg)* ;
arg         : ID '=' value ;
contractField: ID '=' (value | array) ';' ;
array       : '[' (value (',' value)*)? ']' ;
value       : INT | ID | STRING ;

ID          : [a-zA-Z_][a-zA-Z0-9_]* ;
INT         : [0-9]+ ;
STRING      : '"' (~["\r\n])* '"' ;
WS          : [ \t\r\n]+ -> skip ;
COMMENT     : '//' ~[\r\n]* -> skip ;
