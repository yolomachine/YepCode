ò"
Þ
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
9
	IdentityN

input2T
output2T"
T
list(type)(0
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718¾¼

Conv1D_3_128/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameConv1D_3_128/kernel

'Conv1D_3_128/kernel/Read/ReadVariableOpReadVariableOpConv1D_3_128/kernel*$
_output_shapes
:*
dtype0
{
Conv1D_3_128/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameConv1D_3_128/bias
t
%Conv1D_3_128/bias/Read/ReadVariableOpReadVariableOpConv1D_3_128/bias*
_output_shapes	
:*
dtype0

Conv1D_5_128/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameConv1D_5_128/kernel

'Conv1D_5_128/kernel/Read/ReadVariableOpReadVariableOpConv1D_5_128/kernel*$
_output_shapes
:*
dtype0
{
Conv1D_5_128/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameConv1D_5_128/bias
t
%Conv1D_5_128/bias/Read/ReadVariableOpReadVariableOpConv1D_5_128/bias*
_output_shapes	
:*
dtype0

Conv1D_7_128/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameConv1D_7_128/kernel

'Conv1D_7_128/kernel/Read/ReadVariableOpReadVariableOpConv1D_7_128/kernel*$
_output_shapes
:*
dtype0
{
Conv1D_7_128/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameConv1D_7_128/bias
t
%Conv1D_7_128/bias/Read/ReadVariableOpReadVariableOpConv1D_7_128/bias*
_output_shapes	
:*
dtype0

Batch_Norm_3_128/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameBatch_Norm_3_128/gamma
~
*Batch_Norm_3_128/gamma/Read/ReadVariableOpReadVariableOpBatch_Norm_3_128/gamma*
_output_shapes	
:*
dtype0

Batch_Norm_3_128/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameBatch_Norm_3_128/beta
|
)Batch_Norm_3_128/beta/Read/ReadVariableOpReadVariableOpBatch_Norm_3_128/beta*
_output_shapes	
:*
dtype0

Batch_Norm_3_128/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameBatch_Norm_3_128/moving_mean

0Batch_Norm_3_128/moving_mean/Read/ReadVariableOpReadVariableOpBatch_Norm_3_128/moving_mean*
_output_shapes	
:*
dtype0

 Batch_Norm_3_128/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Batch_Norm_3_128/moving_variance

4Batch_Norm_3_128/moving_variance/Read/ReadVariableOpReadVariableOp Batch_Norm_3_128/moving_variance*
_output_shapes	
:*
dtype0

Batch_Norm_5_128/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameBatch_Norm_5_128/gamma
~
*Batch_Norm_5_128/gamma/Read/ReadVariableOpReadVariableOpBatch_Norm_5_128/gamma*
_output_shapes	
:*
dtype0

Batch_Norm_5_128/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameBatch_Norm_5_128/beta
|
)Batch_Norm_5_128/beta/Read/ReadVariableOpReadVariableOpBatch_Norm_5_128/beta*
_output_shapes	
:*
dtype0

Batch_Norm_5_128/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameBatch_Norm_5_128/moving_mean

0Batch_Norm_5_128/moving_mean/Read/ReadVariableOpReadVariableOpBatch_Norm_5_128/moving_mean*
_output_shapes	
:*
dtype0

 Batch_Norm_5_128/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Batch_Norm_5_128/moving_variance

4Batch_Norm_5_128/moving_variance/Read/ReadVariableOpReadVariableOp Batch_Norm_5_128/moving_variance*
_output_shapes	
:*
dtype0

Batch_Norm_7_128/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameBatch_Norm_7_128/gamma
~
*Batch_Norm_7_128/gamma/Read/ReadVariableOpReadVariableOpBatch_Norm_7_128/gamma*
_output_shapes	
:*
dtype0

Batch_Norm_7_128/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameBatch_Norm_7_128/beta
|
)Batch_Norm_7_128/beta/Read/ReadVariableOpReadVariableOpBatch_Norm_7_128/beta*
_output_shapes	
:*
dtype0

Batch_Norm_7_128/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameBatch_Norm_7_128/moving_mean

0Batch_Norm_7_128/moving_mean/Read/ReadVariableOpReadVariableOpBatch_Norm_7_128/moving_mean*
_output_shapes	
:*
dtype0

 Batch_Norm_7_128/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Batch_Norm_7_128/moving_variance

4Batch_Norm_7_128/moving_variance/Read/ReadVariableOpReadVariableOp Batch_Norm_7_128/moving_variance*
_output_shapes	
:*
dtype0

Conv1D_3_256/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameConv1D_3_256/kernel

'Conv1D_3_256/kernel/Read/ReadVariableOpReadVariableOpConv1D_3_256/kernel*$
_output_shapes
:*
dtype0
{
Conv1D_3_256/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameConv1D_3_256/bias
t
%Conv1D_3_256/bias/Read/ReadVariableOpReadVariableOpConv1D_3_256/bias*
_output_shapes	
:*
dtype0

Conv1D_5_256/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameConv1D_5_256/kernel

'Conv1D_5_256/kernel/Read/ReadVariableOpReadVariableOpConv1D_5_256/kernel*$
_output_shapes
:*
dtype0
{
Conv1D_5_256/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameConv1D_5_256/bias
t
%Conv1D_5_256/bias/Read/ReadVariableOpReadVariableOpConv1D_5_256/bias*
_output_shapes	
:*
dtype0

Conv1D_7_256/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameConv1D_7_256/kernel

'Conv1D_7_256/kernel/Read/ReadVariableOpReadVariableOpConv1D_7_256/kernel*$
_output_shapes
:*
dtype0
{
Conv1D_7_256/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameConv1D_7_256/bias
t
%Conv1D_7_256/bias/Read/ReadVariableOpReadVariableOpConv1D_7_256/bias*
_output_shapes	
:*
dtype0

Batch_Norm_3_256/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameBatch_Norm_3_256/gamma
~
*Batch_Norm_3_256/gamma/Read/ReadVariableOpReadVariableOpBatch_Norm_3_256/gamma*
_output_shapes	
:*
dtype0

Batch_Norm_3_256/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameBatch_Norm_3_256/beta
|
)Batch_Norm_3_256/beta/Read/ReadVariableOpReadVariableOpBatch_Norm_3_256/beta*
_output_shapes	
:*
dtype0

Batch_Norm_3_256/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameBatch_Norm_3_256/moving_mean

0Batch_Norm_3_256/moving_mean/Read/ReadVariableOpReadVariableOpBatch_Norm_3_256/moving_mean*
_output_shapes	
:*
dtype0

 Batch_Norm_3_256/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Batch_Norm_3_256/moving_variance

4Batch_Norm_3_256/moving_variance/Read/ReadVariableOpReadVariableOp Batch_Norm_3_256/moving_variance*
_output_shapes	
:*
dtype0

Batch_Norm_5_256/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameBatch_Norm_5_256/gamma
~
*Batch_Norm_5_256/gamma/Read/ReadVariableOpReadVariableOpBatch_Norm_5_256/gamma*
_output_shapes	
:*
dtype0

Batch_Norm_5_256/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameBatch_Norm_5_256/beta
|
)Batch_Norm_5_256/beta/Read/ReadVariableOpReadVariableOpBatch_Norm_5_256/beta*
_output_shapes	
:*
dtype0

Batch_Norm_5_256/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameBatch_Norm_5_256/moving_mean

0Batch_Norm_5_256/moving_mean/Read/ReadVariableOpReadVariableOpBatch_Norm_5_256/moving_mean*
_output_shapes	
:*
dtype0

 Batch_Norm_5_256/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Batch_Norm_5_256/moving_variance

4Batch_Norm_5_256/moving_variance/Read/ReadVariableOpReadVariableOp Batch_Norm_5_256/moving_variance*
_output_shapes	
:*
dtype0

Batch_Norm_7_256/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameBatch_Norm_7_256/gamma
~
*Batch_Norm_7_256/gamma/Read/ReadVariableOpReadVariableOpBatch_Norm_7_256/gamma*
_output_shapes	
:*
dtype0

Batch_Norm_7_256/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameBatch_Norm_7_256/beta
|
)Batch_Norm_7_256/beta/Read/ReadVariableOpReadVariableOpBatch_Norm_7_256/beta*
_output_shapes	
:*
dtype0

Batch_Norm_7_256/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameBatch_Norm_7_256/moving_mean

0Batch_Norm_7_256/moving_mean/Read/ReadVariableOpReadVariableOpBatch_Norm_7_256/moving_mean*
_output_shapes	
:*
dtype0

 Batch_Norm_7_256/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Batch_Norm_7_256/moving_variance

4Batch_Norm_7_256/moving_variance/Read/ReadVariableOpReadVariableOp Batch_Norm_7_256/moving_variance*
_output_shapes	
:*
dtype0

Prediction/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*"
shared_namePrediction/kernel
x
%Prediction/kernel/Read/ReadVariableOpReadVariableOpPrediction/kernel*
_output_shapes
:	*
dtype0
v
Prediction/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namePrediction/bias
o
#Prediction/bias/Read/ReadVariableOpReadVariableOpPrediction/bias*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

NoOpNoOp
çg
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*¢g
valuegBg Bg

layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer_with_weights-7

layer-9
layer_with_weights-8
layer-10
layer_with_weights-9
layer-11
layer_with_weights-10
layer-12
layer_with_weights-11
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer_with_weights-12
layer-19

signatures
#_self_saveable_object_factories
	optimizer
loss
trainable_variables
regularization_losses
	variables
	keras_api
%
#_self_saveable_object_factories
w
#_self_saveable_object_factories
trainable_variables
 regularization_losses
!	variables
"	keras_api


#kernel
$bias
#%_self_saveable_object_factories
&trainable_variables
'regularization_losses
(	variables
)	keras_api


*kernel
+bias
#,_self_saveable_object_factories
-trainable_variables
.regularization_losses
/	variables
0	keras_api


1kernel
2bias
#3_self_saveable_object_factories
4trainable_variables
5regularization_losses
6	variables
7	keras_api
¼
8axis
	9gamma
:beta
;moving_mean
<moving_variance
#=_self_saveable_object_factories
>trainable_variables
?regularization_losses
@	variables
A	keras_api
¼
Baxis
	Cgamma
Dbeta
Emoving_mean
Fmoving_variance
#G_self_saveable_object_factories
Htrainable_variables
Iregularization_losses
J	variables
K	keras_api
¼
Laxis
	Mgamma
Nbeta
Omoving_mean
Pmoving_variance
#Q_self_saveable_object_factories
Rtrainable_variables
Sregularization_losses
T	variables
U	keras_api


Vkernel
Wbias
#X_self_saveable_object_factories
Ytrainable_variables
Zregularization_losses
[	variables
\	keras_api


]kernel
^bias
#__self_saveable_object_factories
`trainable_variables
aregularization_losses
b	variables
c	keras_api


dkernel
ebias
#f_self_saveable_object_factories
gtrainable_variables
hregularization_losses
i	variables
j	keras_api
¼
kaxis
	lgamma
mbeta
nmoving_mean
omoving_variance
#p_self_saveable_object_factories
qtrainable_variables
rregularization_losses
s	variables
t	keras_api
¼
uaxis
	vgamma
wbeta
xmoving_mean
ymoving_variance
#z_self_saveable_object_factories
{trainable_variables
|regularization_losses
}	variables
~	keras_api
Å
axis

gamma
	beta
moving_mean
moving_variance
$_self_saveable_object_factories
trainable_variables
regularization_losses
	variables
	keras_api
|
$_self_saveable_object_factories
trainable_variables
regularization_losses
	variables
	keras_api
|
$_self_saveable_object_factories
trainable_variables
regularization_losses
	variables
	keras_api
|
$_self_saveable_object_factories
trainable_variables
regularization_losses
	variables
	keras_api
|
$_self_saveable_object_factories
trainable_variables
regularization_losses
	variables
	keras_api
|
$_self_saveable_object_factories
trainable_variables
regularization_losses
 	variables
¡	keras_api

¢kernel
	£bias
$¤_self_saveable_object_factories
¥trainable_variables
¦regularization_losses
§	variables
¨	keras_api
 
 
 
 
Ê
#0
$1
*2
+3
14
25
96
:7
C8
D9
M10
N11
V12
W13
]14
^15
d16
e17
l18
m19
v20
w21
22
23
¢24
£25
 
¬
#0
$1
*2
+3
14
25
96
:7
;8
<9
C10
D11
E12
F13
M14
N15
O16
P17
V18
W19
]20
^21
d22
e23
l24
m25
n26
o27
v28
w29
x30
y31
32
33
34
35
¢36
£37
²
trainable_variables
©non_trainable_variables
 ªlayer_regularization_losses
regularization_losses
	variables
«metrics
¬layers
­layer_metrics
 
 
 
 
 
²
trainable_variables
®non_trainable_variables
 ¯layer_regularization_losses
 regularization_losses
!	variables
°metrics
±layers
²layer_metrics
_]
VARIABLE_VALUEConv1D_3_128/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEConv1D_3_128/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

#0
$1
 

#0
$1
²
&trainable_variables
³non_trainable_variables
 ´layer_regularization_losses
'regularization_losses
(	variables
µmetrics
¶layers
·layer_metrics
_]
VARIABLE_VALUEConv1D_5_128/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEConv1D_5_128/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

*0
+1
 

*0
+1
²
-trainable_variables
¸non_trainable_variables
 ¹layer_regularization_losses
.regularization_losses
/	variables
ºmetrics
»layers
¼layer_metrics
_]
VARIABLE_VALUEConv1D_7_128/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEConv1D_7_128/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

10
21
 

10
21
²
4trainable_variables
½non_trainable_variables
 ¾layer_regularization_losses
5regularization_losses
6	variables
¿metrics
Àlayers
Álayer_metrics
 
a_
VARIABLE_VALUEBatch_Norm_3_128/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEBatch_Norm_3_128/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEBatch_Norm_3_128/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE Batch_Norm_3_128/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

90
:1
 

90
:1
;2
<3
²
>trainable_variables
Ânon_trainable_variables
 Ãlayer_regularization_losses
?regularization_losses
@	variables
Ämetrics
Ålayers
Ælayer_metrics
 
a_
VARIABLE_VALUEBatch_Norm_5_128/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEBatch_Norm_5_128/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEBatch_Norm_5_128/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE Batch_Norm_5_128/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

C0
D1
 

C0
D1
E2
F3
²
Htrainable_variables
Çnon_trainable_variables
 Èlayer_regularization_losses
Iregularization_losses
J	variables
Émetrics
Êlayers
Ëlayer_metrics
 
a_
VARIABLE_VALUEBatch_Norm_7_128/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEBatch_Norm_7_128/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEBatch_Norm_7_128/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE Batch_Norm_7_128/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

M0
N1
 

M0
N1
O2
P3
²
Rtrainable_variables
Ìnon_trainable_variables
 Ílayer_regularization_losses
Sregularization_losses
T	variables
Îmetrics
Ïlayers
Ðlayer_metrics
_]
VARIABLE_VALUEConv1D_3_256/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEConv1D_3_256/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

V0
W1
 

V0
W1
²
Ytrainable_variables
Ñnon_trainable_variables
 Òlayer_regularization_losses
Zregularization_losses
[	variables
Ómetrics
Ôlayers
Õlayer_metrics
_]
VARIABLE_VALUEConv1D_5_256/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEConv1D_5_256/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

]0
^1
 

]0
^1
²
`trainable_variables
Önon_trainable_variables
 ×layer_regularization_losses
aregularization_losses
b	variables
Ømetrics
Ùlayers
Úlayer_metrics
_]
VARIABLE_VALUEConv1D_7_256/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEConv1D_7_256/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 

d0
e1
 

d0
e1
²
gtrainable_variables
Ûnon_trainable_variables
 Ülayer_regularization_losses
hregularization_losses
i	variables
Ýmetrics
Þlayers
ßlayer_metrics
 
a_
VARIABLE_VALUEBatch_Norm_3_256/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEBatch_Norm_3_256/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEBatch_Norm_3_256/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE Batch_Norm_3_256/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

l0
m1
 

l0
m1
n2
o3
²
qtrainable_variables
ànon_trainable_variables
 álayer_regularization_losses
rregularization_losses
s	variables
âmetrics
ãlayers
älayer_metrics
 
b`
VARIABLE_VALUEBatch_Norm_5_256/gamma6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEBatch_Norm_5_256/beta5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEBatch_Norm_5_256/moving_mean<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE Batch_Norm_5_256/moving_variance@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

v0
w1
 

v0
w1
x2
y3
²
{trainable_variables
ånon_trainable_variables
 ælayer_regularization_losses
|regularization_losses
}	variables
çmetrics
èlayers
élayer_metrics
 
b`
VARIABLE_VALUEBatch_Norm_7_256/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEBatch_Norm_7_256/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEBatch_Norm_7_256/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE Batch_Norm_7_256/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
 
 
0
1
2
3
µ
trainable_variables
ênon_trainable_variables
 ëlayer_regularization_losses
regularization_losses
	variables
ìmetrics
ílayers
îlayer_metrics
 
 
 
 
µ
trainable_variables
ïnon_trainable_variables
 ðlayer_regularization_losses
regularization_losses
	variables
ñmetrics
òlayers
ólayer_metrics
 
 
 
 
µ
trainable_variables
ônon_trainable_variables
 õlayer_regularization_losses
regularization_losses
	variables
ömetrics
÷layers
ølayer_metrics
 
 
 
 
µ
trainable_variables
ùnon_trainable_variables
 úlayer_regularization_losses
regularization_losses
	variables
ûmetrics
ülayers
ýlayer_metrics
 
 
 
 
µ
trainable_variables
þnon_trainable_variables
 ÿlayer_regularization_losses
regularization_losses
	variables
metrics
layers
layer_metrics
 
 
 
 
µ
trainable_variables
non_trainable_variables
 layer_regularization_losses
regularization_losses
 	variables
metrics
layers
layer_metrics
^\
VARIABLE_VALUEPrediction/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEPrediction/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE
 

¢0
£1
 

¢0
£1
µ
¥trainable_variables
non_trainable_variables
 layer_regularization_losses
¦regularization_losses
§	variables
metrics
layers
layer_metrics
X
;0
<1
E2
F3
O4
P5
n6
o7
x8
y9
10
11
 

0

0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

;0
<1
 
 
 
 

E0
F1
 
 
 
 

O0
P1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

n0
o1
 
 
 
 

x0
y1
 
 
 
 

0
1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

total

count
	variables
	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

0
1

	variables

serving_default_InputsPlaceholder*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype0**
shape!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
²

StatefulPartitionedCallStatefulPartitionedCallserving_default_InputsConv1D_7_128/kernelConv1D_7_128/biasConv1D_5_128/kernelConv1D_5_128/biasConv1D_3_128/kernelConv1D_3_128/bias Batch_Norm_7_128/moving_varianceBatch_Norm_7_128/gammaBatch_Norm_7_128/moving_meanBatch_Norm_7_128/beta Batch_Norm_5_128/moving_varianceBatch_Norm_5_128/gammaBatch_Norm_5_128/moving_meanBatch_Norm_5_128/beta Batch_Norm_3_128/moving_varianceBatch_Norm_3_128/gammaBatch_Norm_3_128/moving_meanBatch_Norm_3_128/betaConv1D_7_256/kernelConv1D_7_256/biasConv1D_5_256/kernelConv1D_5_256/biasConv1D_3_256/kernelConv1D_3_256/bias Batch_Norm_7_256/moving_varianceBatch_Norm_7_256/gammaBatch_Norm_7_256/moving_meanBatch_Norm_7_256/beta Batch_Norm_5_256/moving_varianceBatch_Norm_5_256/gammaBatch_Norm_5_256/moving_meanBatch_Norm_5_256/beta Batch_Norm_3_256/moving_varianceBatch_Norm_3_256/gammaBatch_Norm_3_256/moving_meanBatch_Norm_3_256/betaPrediction/kernelPrediction/bias*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_signature_wrapper_1916623629
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
¦
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename'Conv1D_3_128/kernel/Read/ReadVariableOp%Conv1D_3_128/bias/Read/ReadVariableOp'Conv1D_5_128/kernel/Read/ReadVariableOp%Conv1D_5_128/bias/Read/ReadVariableOp'Conv1D_7_128/kernel/Read/ReadVariableOp%Conv1D_7_128/bias/Read/ReadVariableOp*Batch_Norm_3_128/gamma/Read/ReadVariableOp)Batch_Norm_3_128/beta/Read/ReadVariableOp0Batch_Norm_3_128/moving_mean/Read/ReadVariableOp4Batch_Norm_3_128/moving_variance/Read/ReadVariableOp*Batch_Norm_5_128/gamma/Read/ReadVariableOp)Batch_Norm_5_128/beta/Read/ReadVariableOp0Batch_Norm_5_128/moving_mean/Read/ReadVariableOp4Batch_Norm_5_128/moving_variance/Read/ReadVariableOp*Batch_Norm_7_128/gamma/Read/ReadVariableOp)Batch_Norm_7_128/beta/Read/ReadVariableOp0Batch_Norm_7_128/moving_mean/Read/ReadVariableOp4Batch_Norm_7_128/moving_variance/Read/ReadVariableOp'Conv1D_3_256/kernel/Read/ReadVariableOp%Conv1D_3_256/bias/Read/ReadVariableOp'Conv1D_5_256/kernel/Read/ReadVariableOp%Conv1D_5_256/bias/Read/ReadVariableOp'Conv1D_7_256/kernel/Read/ReadVariableOp%Conv1D_7_256/bias/Read/ReadVariableOp*Batch_Norm_3_256/gamma/Read/ReadVariableOp)Batch_Norm_3_256/beta/Read/ReadVariableOp0Batch_Norm_3_256/moving_mean/Read/ReadVariableOp4Batch_Norm_3_256/moving_variance/Read/ReadVariableOp*Batch_Norm_5_256/gamma/Read/ReadVariableOp)Batch_Norm_5_256/beta/Read/ReadVariableOp0Batch_Norm_5_256/moving_mean/Read/ReadVariableOp4Batch_Norm_5_256/moving_variance/Read/ReadVariableOp*Batch_Norm_7_256/gamma/Read/ReadVariableOp)Batch_Norm_7_256/beta/Read/ReadVariableOp0Batch_Norm_7_256/moving_mean/Read/ReadVariableOp4Batch_Norm_7_256/moving_variance/Read/ReadVariableOp%Prediction/kernel/Read/ReadVariableOp#Prediction/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*5
Tin.
,2**
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference__traced_save_1916625219


StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameConv1D_3_128/kernelConv1D_3_128/biasConv1D_5_128/kernelConv1D_5_128/biasConv1D_7_128/kernelConv1D_7_128/biasBatch_Norm_3_128/gammaBatch_Norm_3_128/betaBatch_Norm_3_128/moving_mean Batch_Norm_3_128/moving_varianceBatch_Norm_5_128/gammaBatch_Norm_5_128/betaBatch_Norm_5_128/moving_mean Batch_Norm_5_128/moving_varianceBatch_Norm_7_128/gammaBatch_Norm_7_128/betaBatch_Norm_7_128/moving_mean Batch_Norm_7_128/moving_varianceConv1D_3_256/kernelConv1D_3_256/biasConv1D_5_256/kernelConv1D_5_256/biasConv1D_7_256/kernelConv1D_7_256/biasBatch_Norm_3_256/gammaBatch_Norm_3_256/betaBatch_Norm_3_256/moving_mean Batch_Norm_3_256/moving_varianceBatch_Norm_5_256/gammaBatch_Norm_5_256/betaBatch_Norm_5_256/moving_mean Batch_Norm_5_256/moving_varianceBatch_Norm_7_256/gammaBatch_Norm_7_256/betaBatch_Norm_7_256/moving_mean Batch_Norm_7_256/moving_variancePrediction/kernelPrediction/biastotalcount*4
Tin-
+2)*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 */
f*R(
&__inference__traced_restore_1916625349ÚÚ
ü*
í
P__inference_Batch_Norm_5_256_layer_call_and_return_conditional_losses_1916624934

inputs6
'assignmovingavg_readvariableop_resource:	8
)assignmovingavg_1_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	0
!batchnorm_readvariableop_resource:	
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:2
moments/StopGradient²
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices·
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay¥
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg/mul¿
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay«
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp¡
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¼
Ô
5__inference_Batch_Norm_7_256_layer_call_fn_1916624960

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_Batch_Norm_7_256_layer_call_and_return_conditional_losses_19166224182
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¸

ü
J__inference_Prediction_layer_call_and_return_conditional_losses_1916625076

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoid
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­

/__inference_Prediction_layer_call_fn_1916625065

inputs
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Prediction_layer_call_and_return_conditional_losses_19166227962
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü
 
L__inference_Conv1D_5_128_layer_call_and_return_conditional_losses_1916624414

inputsC
+conv1d_expanddims_1_readvariableop_resource:.
biasadd_readvariableop_resource:	

identity_1¢BiasAdd/ReadVariableOp¢"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims/dim 
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
conv1d/ExpandDimsº
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim¹
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv1d/ExpandDims_1À
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAddo
SigmoidSigmoidBiasAdd:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
Sigmoidp
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
muli
IdentityIdentitymul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

IdentityÕ
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1916624407*V
_output_shapesD
B:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	IdentityN¶

Identity_1IdentityIdentityN:output:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ë«

&__inference__traced_restore_1916625349
file_prefix<
$assignvariableop_conv1d_3_128_kernel:3
$assignvariableop_1_conv1d_3_128_bias:	>
&assignvariableop_2_conv1d_5_128_kernel:3
$assignvariableop_3_conv1d_5_128_bias:	>
&assignvariableop_4_conv1d_7_128_kernel:3
$assignvariableop_5_conv1d_7_128_bias:	8
)assignvariableop_6_batch_norm_3_128_gamma:	7
(assignvariableop_7_batch_norm_3_128_beta:	>
/assignvariableop_8_batch_norm_3_128_moving_mean:	B
3assignvariableop_9_batch_norm_3_128_moving_variance:	9
*assignvariableop_10_batch_norm_5_128_gamma:	8
)assignvariableop_11_batch_norm_5_128_beta:	?
0assignvariableop_12_batch_norm_5_128_moving_mean:	C
4assignvariableop_13_batch_norm_5_128_moving_variance:	9
*assignvariableop_14_batch_norm_7_128_gamma:	8
)assignvariableop_15_batch_norm_7_128_beta:	?
0assignvariableop_16_batch_norm_7_128_moving_mean:	C
4assignvariableop_17_batch_norm_7_128_moving_variance:	?
'assignvariableop_18_conv1d_3_256_kernel:4
%assignvariableop_19_conv1d_3_256_bias:	?
'assignvariableop_20_conv1d_5_256_kernel:4
%assignvariableop_21_conv1d_5_256_bias:	?
'assignvariableop_22_conv1d_7_256_kernel:4
%assignvariableop_23_conv1d_7_256_bias:	9
*assignvariableop_24_batch_norm_3_256_gamma:	8
)assignvariableop_25_batch_norm_3_256_beta:	?
0assignvariableop_26_batch_norm_3_256_moving_mean:	C
4assignvariableop_27_batch_norm_3_256_moving_variance:	9
*assignvariableop_28_batch_norm_5_256_gamma:	8
)assignvariableop_29_batch_norm_5_256_beta:	?
0assignvariableop_30_batch_norm_5_256_moving_mean:	C
4assignvariableop_31_batch_norm_5_256_moving_variance:	9
*assignvariableop_32_batch_norm_7_256_gamma:	8
)assignvariableop_33_batch_norm_7_256_beta:	?
0assignvariableop_34_batch_norm_7_256_moving_mean:	C
4assignvariableop_35_batch_norm_7_256_moving_variance:	8
%assignvariableop_36_prediction_kernel:	1
#assignvariableop_37_prediction_bias:#
assignvariableop_38_total: #
assignvariableop_39_count: 
identity_41¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9¡
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*­
value£B )B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesà
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*e
value\BZ)B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesû
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*º
_output_shapes§
¤:::::::::::::::::::::::::::::::::::::::::*7
dtypes-
+2)2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity£
AssignVariableOpAssignVariableOp$assignvariableop_conv1d_3_128_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1©
AssignVariableOp_1AssignVariableOp$assignvariableop_1_conv1d_3_128_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2«
AssignVariableOp_2AssignVariableOp&assignvariableop_2_conv1d_5_128_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3©
AssignVariableOp_3AssignVariableOp$assignvariableop_3_conv1d_5_128_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4«
AssignVariableOp_4AssignVariableOp&assignvariableop_4_conv1d_7_128_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5©
AssignVariableOp_5AssignVariableOp$assignvariableop_5_conv1d_7_128_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6®
AssignVariableOp_6AssignVariableOp)assignvariableop_6_batch_norm_3_128_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7­
AssignVariableOp_7AssignVariableOp(assignvariableop_7_batch_norm_3_128_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8´
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_norm_3_128_moving_meanIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¸
AssignVariableOp_9AssignVariableOp3assignvariableop_9_batch_norm_3_128_moving_varianceIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10²
AssignVariableOp_10AssignVariableOp*assignvariableop_10_batch_norm_5_128_gammaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11±
AssignVariableOp_11AssignVariableOp)assignvariableop_11_batch_norm_5_128_betaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12¸
AssignVariableOp_12AssignVariableOp0assignvariableop_12_batch_norm_5_128_moving_meanIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13¼
AssignVariableOp_13AssignVariableOp4assignvariableop_13_batch_norm_5_128_moving_varianceIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14²
AssignVariableOp_14AssignVariableOp*assignvariableop_14_batch_norm_7_128_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15±
AssignVariableOp_15AssignVariableOp)assignvariableop_15_batch_norm_7_128_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16¸
AssignVariableOp_16AssignVariableOp0assignvariableop_16_batch_norm_7_128_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17¼
AssignVariableOp_17AssignVariableOp4assignvariableop_17_batch_norm_7_128_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18¯
AssignVariableOp_18AssignVariableOp'assignvariableop_18_conv1d_3_256_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19­
AssignVariableOp_19AssignVariableOp%assignvariableop_19_conv1d_3_256_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20¯
AssignVariableOp_20AssignVariableOp'assignvariableop_20_conv1d_5_256_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21­
AssignVariableOp_21AssignVariableOp%assignvariableop_21_conv1d_5_256_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22¯
AssignVariableOp_22AssignVariableOp'assignvariableop_22_conv1d_7_256_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23­
AssignVariableOp_23AssignVariableOp%assignvariableop_23_conv1d_7_256_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24²
AssignVariableOp_24AssignVariableOp*assignvariableop_24_batch_norm_3_256_gammaIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25±
AssignVariableOp_25AssignVariableOp)assignvariableop_25_batch_norm_3_256_betaIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26¸
AssignVariableOp_26AssignVariableOp0assignvariableop_26_batch_norm_3_256_moving_meanIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27¼
AssignVariableOp_27AssignVariableOp4assignvariableop_27_batch_norm_3_256_moving_varianceIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28²
AssignVariableOp_28AssignVariableOp*assignvariableop_28_batch_norm_5_256_gammaIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29±
AssignVariableOp_29AssignVariableOp)assignvariableop_29_batch_norm_5_256_betaIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30¸
AssignVariableOp_30AssignVariableOp0assignvariableop_30_batch_norm_5_256_moving_meanIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31¼
AssignVariableOp_31AssignVariableOp4assignvariableop_31_batch_norm_5_256_moving_varianceIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32²
AssignVariableOp_32AssignVariableOp*assignvariableop_32_batch_norm_7_256_gammaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33±
AssignVariableOp_33AssignVariableOp)assignvariableop_33_batch_norm_7_256_betaIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34¸
AssignVariableOp_34AssignVariableOp0assignvariableop_34_batch_norm_7_256_moving_meanIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35¼
AssignVariableOp_35AssignVariableOp4assignvariableop_35_batch_norm_7_256_moving_varianceIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36­
AssignVariableOp_36AssignVariableOp%assignvariableop_36_prediction_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37«
AssignVariableOp_37AssignVariableOp#assignvariableop_37_prediction_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38¡
AssignVariableOp_38AssignVariableOpassignvariableop_38_totalIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39¡
AssignVariableOp_39AssignVariableOpassignvariableop_39_countIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_399
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpÎ
Identity_40Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_40Á
Identity_41IdentityIdentity_40:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_41"#
identity_41Identity_41:output:0*e
_input_shapesT
R: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ü
 
L__inference_Conv1D_7_128_layer_call_and_return_conditional_losses_1916622570

inputsC
+conv1d_expanddims_1_readvariableop_resource:.
biasadd_readvariableop_resource:	

identity_1¢BiasAdd/ReadVariableOp¢"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims/dim 
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
conv1d/ExpandDimsº
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim¹
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv1d/ExpandDims_1À
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAddo
SigmoidSigmoidBiasAdd:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
Sigmoidp
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
muli
IdentityIdentitymul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

IdentityÕ
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1916622563*V
_output_shapesD
B:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	IdentityN¶

Identity_1IdentityIdentityN:output:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ
³
P__inference_Batch_Norm_3_256_layer_call_and_return_conditional_losses_1916622034

inputs0
!batchnorm_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	2
#batchnorm_readvariableop_1_resource:	2
#batchnorm_readvariableop_2_resource:	
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1é
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
á
T
8__inference_Concatenate_Dropout_layer_call_fn_1916625034

inputs
identityÕ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_Concatenate_Dropout_layer_call_and_return_conditional_losses_19166227832
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ
³
P__inference_Batch_Norm_5_128_layer_call_and_return_conditional_losses_1916624570

inputs0
!batchnorm_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	2
#batchnorm_readvariableop_1_resource:	2
#batchnorm_readvariableop_2_resource:	
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1é
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

q
S__inference_Concatenate_Dropout_layer_call_and_return_conditional_losses_1916625044

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ
³
P__inference_Batch_Norm_5_128_layer_call_and_return_conditional_losses_1916621710

inputs0
!batchnorm_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	2
#batchnorm_readvariableop_1_resource:	2
#batchnorm_readvariableop_2_resource:	
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1é
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

R
6__inference_Embedding_Dropout_layer_call_fn_1916624332

inputs
identityà
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_Embedding_Dropout_layer_call_and_return_conditional_losses_19166225472
PartitionedCallz
IdentityIdentityPartitionedCall:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü*
í
P__inference_Batch_Norm_3_256_layer_call_and_return_conditional_losses_1916624854

inputs6
'assignmovingavg_readvariableop_resource:	8
)assignmovingavg_1_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	0
!batchnorm_readvariableop_resource:	
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:2
moments/StopGradient²
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices·
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay¥
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg/mul¿
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay«
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp¡
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü
 
L__inference_Conv1D_3_256_layer_call_and_return_conditional_losses_1916624714

inputsC
+conv1d_expanddims_1_readvariableop_resource:.
biasadd_readvariableop_resource:	

identity_1¢BiasAdd/ReadVariableOp¢"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims/dim 
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
conv1d/ExpandDimsº
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim¹
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv1d/ExpandDims_1À
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAddo
SigmoidSigmoidBiasAdd:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
Sigmoidp
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
muli
IdentityIdentitymul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

IdentityÕ
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1916624707*V
_output_shapesD
B:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	IdentityN¶

Identity_1IdentityIdentityN:output:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ
³
P__inference_Batch_Norm_7_256_layer_call_and_return_conditional_losses_1916622358

inputs0
!batchnorm_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	2
#batchnorm_readvariableop_1_resource:	2
#batchnorm_readvariableop_2_resource:	
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1é
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÈT

#__inference__traced_save_1916625219
file_prefix2
.savev2_conv1d_3_128_kernel_read_readvariableop0
,savev2_conv1d_3_128_bias_read_readvariableop2
.savev2_conv1d_5_128_kernel_read_readvariableop0
,savev2_conv1d_5_128_bias_read_readvariableop2
.savev2_conv1d_7_128_kernel_read_readvariableop0
,savev2_conv1d_7_128_bias_read_readvariableop5
1savev2_batch_norm_3_128_gamma_read_readvariableop4
0savev2_batch_norm_3_128_beta_read_readvariableop;
7savev2_batch_norm_3_128_moving_mean_read_readvariableop?
;savev2_batch_norm_3_128_moving_variance_read_readvariableop5
1savev2_batch_norm_5_128_gamma_read_readvariableop4
0savev2_batch_norm_5_128_beta_read_readvariableop;
7savev2_batch_norm_5_128_moving_mean_read_readvariableop?
;savev2_batch_norm_5_128_moving_variance_read_readvariableop5
1savev2_batch_norm_7_128_gamma_read_readvariableop4
0savev2_batch_norm_7_128_beta_read_readvariableop;
7savev2_batch_norm_7_128_moving_mean_read_readvariableop?
;savev2_batch_norm_7_128_moving_variance_read_readvariableop2
.savev2_conv1d_3_256_kernel_read_readvariableop0
,savev2_conv1d_3_256_bias_read_readvariableop2
.savev2_conv1d_5_256_kernel_read_readvariableop0
,savev2_conv1d_5_256_bias_read_readvariableop2
.savev2_conv1d_7_256_kernel_read_readvariableop0
,savev2_conv1d_7_256_bias_read_readvariableop5
1savev2_batch_norm_3_256_gamma_read_readvariableop4
0savev2_batch_norm_3_256_beta_read_readvariableop;
7savev2_batch_norm_3_256_moving_mean_read_readvariableop?
;savev2_batch_norm_3_256_moving_variance_read_readvariableop5
1savev2_batch_norm_5_256_gamma_read_readvariableop4
0savev2_batch_norm_5_256_beta_read_readvariableop;
7savev2_batch_norm_5_256_moving_mean_read_readvariableop?
;savev2_batch_norm_5_256_moving_variance_read_readvariableop5
1savev2_batch_norm_7_256_gamma_read_readvariableop4
0savev2_batch_norm_7_256_beta_read_readvariableop;
7savev2_batch_norm_7_256_moving_mean_read_readvariableop?
;savev2_batch_norm_7_256_moving_variance_read_readvariableop0
,savev2_prediction_kernel_read_readvariableop.
*savev2_prediction_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*­
value£B )B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesÚ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*e
value\BZ)B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesÖ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0.savev2_conv1d_3_128_kernel_read_readvariableop,savev2_conv1d_3_128_bias_read_readvariableop.savev2_conv1d_5_128_kernel_read_readvariableop,savev2_conv1d_5_128_bias_read_readvariableop.savev2_conv1d_7_128_kernel_read_readvariableop,savev2_conv1d_7_128_bias_read_readvariableop1savev2_batch_norm_3_128_gamma_read_readvariableop0savev2_batch_norm_3_128_beta_read_readvariableop7savev2_batch_norm_3_128_moving_mean_read_readvariableop;savev2_batch_norm_3_128_moving_variance_read_readvariableop1savev2_batch_norm_5_128_gamma_read_readvariableop0savev2_batch_norm_5_128_beta_read_readvariableop7savev2_batch_norm_5_128_moving_mean_read_readvariableop;savev2_batch_norm_5_128_moving_variance_read_readvariableop1savev2_batch_norm_7_128_gamma_read_readvariableop0savev2_batch_norm_7_128_beta_read_readvariableop7savev2_batch_norm_7_128_moving_mean_read_readvariableop;savev2_batch_norm_7_128_moving_variance_read_readvariableop.savev2_conv1d_3_256_kernel_read_readvariableop,savev2_conv1d_3_256_bias_read_readvariableop.savev2_conv1d_5_256_kernel_read_readvariableop,savev2_conv1d_5_256_bias_read_readvariableop.savev2_conv1d_7_256_kernel_read_readvariableop,savev2_conv1d_7_256_bias_read_readvariableop1savev2_batch_norm_3_256_gamma_read_readvariableop0savev2_batch_norm_3_256_beta_read_readvariableop7savev2_batch_norm_3_256_moving_mean_read_readvariableop;savev2_batch_norm_3_256_moving_variance_read_readvariableop1savev2_batch_norm_5_256_gamma_read_readvariableop0savev2_batch_norm_5_256_beta_read_readvariableop7savev2_batch_norm_5_256_moving_mean_read_readvariableop;savev2_batch_norm_5_256_moving_variance_read_readvariableop1savev2_batch_norm_7_256_gamma_read_readvariableop0savev2_batch_norm_7_256_beta_read_readvariableop7savev2_batch_norm_7_256_moving_mean_read_readvariableop;savev2_batch_norm_7_256_moving_variance_read_readvariableop,savev2_prediction_kernel_read_readvariableop*savev2_prediction_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *7
dtypes-
+2)2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*à
_input_shapesÎ
Ë: :::::::::::::::::::::::::::::::::::::	:: : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:*&
$
_output_shapes
::!

_output_shapes	
::*&
$
_output_shapes
::!

_output_shapes	
::*&
$
_output_shapes
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!	

_output_shapes	
::!


_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::*&
$
_output_shapes
::!

_output_shapes	
::*&
$
_output_shapes
::!

_output_shapes	
::*&
$
_output_shapes
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::! 

_output_shapes	
::!!

_output_shapes	
::!"

_output_shapes	
::!#

_output_shapes	
::!$

_output_shapes	
::%%!

_output_shapes
:	: &

_output_shapes
::'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: 
¶
o
Q__inference_Embedding_Dropout_layer_call_and_return_conditional_losses_1916622547

inputs

identity_1h
IdentityIdentityinputs*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identityw

Identity_1IdentityIdentity:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í
¥
1__inference_Conv1D_3_128_layer_call_fn_1916624363

inputs
unknown:
	unknown_0:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_Conv1D_3_128_layer_call_and_return_conditional_losses_19166226242
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ð
	
1__inference_Tree_Encoder_layer_call_fn_1916623791

inputs
unknown:
	unknown_0:	!
	unknown_1:
	unknown_2:	!
	unknown_3:
	unknown_4:	
	unknown_5:	
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:	

unknown_11:	

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	"

unknown_17:

unknown_18:	"

unknown_19:

unknown_20:	"

unknown_21:

unknown_22:	

unknown_23:	

unknown_24:	

unknown_25:	

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:	

unknown_30:	

unknown_31:	

unknown_32:	

unknown_33:	

unknown_34:	

unknown_35:	

unknown_36:
identity¢StatefulPartitionedCallâ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*<
_read_only_resource_inputs
	
 #$%&*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_Tree_Encoder_layer_call_and_return_conditional_losses_19166231882
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapeso
m:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
©
f
J__inference_Max_Pool_7_layer_call_and_return_conditional_losses_1916622529

inputs
identityp
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Max/reduction_indicest
MaxMaxinputsMax/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Maxi
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¼
Ô
5__inference_Batch_Norm_3_256_layer_call_fn_1916624800

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_Batch_Norm_3_256_layer_call_and_return_conditional_losses_19166220942
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü*
í
P__inference_Batch_Norm_7_256_layer_call_and_return_conditional_losses_1916625014

inputs6
'assignmovingavg_readvariableop_resource:	8
)assignmovingavg_1_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	0
!batchnorm_readvariableop_resource:	
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:2
moments/StopGradient²
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices·
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay¥
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg/mul¿
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay«
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp¡
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ
³
P__inference_Batch_Norm_3_128_layer_call_and_return_conditional_losses_1916621548

inputs0
!batchnorm_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	2
#batchnorm_readvariableop_1_resource:	2
#batchnorm_readvariableop_2_resource:	
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1é
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í
q
8__inference_Concatenate_Dropout_layer_call_fn_1916625039

inputs
identity¢StatefulPartitionedCallí
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_Concatenate_Dropout_layer_call_and_return_conditional_losses_19166229122
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü*
í
P__inference_Batch_Norm_7_128_layer_call_and_return_conditional_losses_1916621932

inputs6
'assignmovingavg_readvariableop_resource:	8
)assignmovingavg_1_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	0
!batchnorm_readvariableop_resource:	
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:2
moments/StopGradient²
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices·
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay¥
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg/mul¿
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay«
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp¡
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

o
6__inference_Embedding_Dropout_layer_call_fn_1916624337

inputs
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_Embedding_Dropout_layer_call_and_return_conditional_losses_19166230032
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü
 
L__inference_Conv1D_5_256_layer_call_and_return_conditional_losses_1916622705

inputsC
+conv1d_expanddims_1_readvariableop_resource:.
biasadd_readvariableop_resource:	

identity_1¢BiasAdd/ReadVariableOp¢"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims/dim 
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
conv1d/ExpandDimsº
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim¹
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv1d/ExpandDims_1À
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAddo
SigmoidSigmoidBiasAdd:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
Sigmoidp
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
muli
IdentityIdentitymul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

IdentityÕ
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1916622698*V
_output_shapesD
B:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	IdentityN¶

Identity_1IdentityIdentityN:output:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¼
Ô
5__inference_Batch_Norm_5_256_layer_call_fn_1916624880

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_Batch_Norm_5_256_layer_call_and_return_conditional_losses_19166222562
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü
 
L__inference_Conv1D_5_256_layer_call_and_return_conditional_losses_1916624744

inputsC
+conv1d_expanddims_1_readvariableop_resource:.
biasadd_readvariableop_resource:	

identity_1¢BiasAdd/ReadVariableOp¢"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims/dim 
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
conv1d/ExpandDimsº
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim¹
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv1d/ExpandDims_1À
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAddo
SigmoidSigmoidBiasAdd:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
Sigmoidp
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
muli
IdentityIdentitymul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

IdentityÕ
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1916624737*V
_output_shapesD
B:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	IdentityN¶

Identity_1IdentityIdentityN:output:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾
Ô
5__inference_Batch_Norm_3_128_layer_call_fn_1916624457

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_Batch_Norm_3_128_layer_call_and_return_conditional_losses_19166215482
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ
³
P__inference_Batch_Norm_7_128_layer_call_and_return_conditional_losses_1916621872

inputs0
!batchnorm_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	2
#batchnorm_readvariableop_1_resource:	2
#batchnorm_readvariableop_2_resource:	
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1é
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü*
í
P__inference_Batch_Norm_3_256_layer_call_and_return_conditional_losses_1916622094

inputs6
'assignmovingavg_readvariableop_resource:	8
)assignmovingavg_1_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	0
!batchnorm_readvariableop_resource:	
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:2
moments/StopGradient²
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices·
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay¥
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg/mul¿
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay«
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp¡
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¼
Ô
5__inference_Batch_Norm_3_128_layer_call_fn_1916624470

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_Batch_Norm_3_128_layer_call_and_return_conditional_losses_19166216082
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü
	
1__inference_Tree_Encoder_layer_call_fn_1916623710

inputs
unknown:
	unknown_0:	!
	unknown_1:
	unknown_2:	!
	unknown_3:
	unknown_4:	
	unknown_5:	
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:	

unknown_11:	

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	"

unknown_17:

unknown_18:	"

unknown_19:

unknown_20:	"

unknown_21:

unknown_22:	

unknown_23:	

unknown_24:	

unknown_25:	

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:	

unknown_30:	

unknown_31:	

unknown_32:	

unknown_33:	

unknown_34:	

unknown_35:	

unknown_36:
identity¢StatefulPartitionedCallî
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_Tree_Encoder_layer_call_and_return_conditional_losses_19166228032
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapeso
m:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü
	
1__inference_Tree_Encoder_layer_call_fn_1916622882

inputs
unknown:
	unknown_0:	!
	unknown_1:
	unknown_2:	!
	unknown_3:
	unknown_4:	
	unknown_5:	
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:	

unknown_11:	

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	"

unknown_17:

unknown_18:	"

unknown_19:

unknown_20:	"

unknown_21:

unknown_22:	

unknown_23:	

unknown_24:	

unknown_25:	

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:	

unknown_30:	

unknown_31:	

unknown_32:	

unknown_33:	

unknown_34:	

unknown_35:	

unknown_36:
identity¢StatefulPartitionedCallî
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_Tree_Encoder_layer_call_and_return_conditional_losses_19166228032
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapeso
m:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameInputs
ü
 
L__inference_Conv1D_3_256_layer_call_and_return_conditional_losses_1916622732

inputsC
+conv1d_expanddims_1_readvariableop_resource:.
biasadd_readvariableop_resource:	

identity_1¢BiasAdd/ReadVariableOp¢"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims/dim 
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
conv1d/ExpandDimsº
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim¹
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv1d/ExpandDims_1À
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAddo
SigmoidSigmoidBiasAdd:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
Sigmoidp
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
muli
IdentityIdentitymul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

IdentityÕ
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1916622725*V
_output_shapesD
B:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	IdentityN¶

Identity_1IdentityIdentityN:output:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

K
/__inference_Max_Pool_5_layer_call_fn_1916622522

inputs
identityÔ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Max_Pool_5_layer_call_and_return_conditional_losses_19166225162
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¶
o
Q__inference_Embedding_Dropout_layer_call_and_return_conditional_losses_1916624342

inputs

identity_1h
IdentityIdentityinputs*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identityw

Identity_1IdentityIdentity:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¼
Ô
5__inference_Batch_Norm_5_128_layer_call_fn_1916624550

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_Batch_Norm_5_128_layer_call_and_return_conditional_losses_19166217702
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾
Ô
5__inference_Batch_Norm_5_128_layer_call_fn_1916624537

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_Batch_Norm_5_128_layer_call_and_return_conditional_losses_19166217102
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í
¥
1__inference_Conv1D_7_256_layer_call_fn_1916624753

inputs
unknown:
	unknown_0:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_Conv1D_7_256_layer_call_and_return_conditional_losses_19166226782
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ
³
P__inference_Batch_Norm_5_256_layer_call_and_return_conditional_losses_1916624900

inputs0
!batchnorm_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	2
#batchnorm_readvariableop_1_resource:	2
#batchnorm_readvariableop_2_resource:	
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1é
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾
Ô
5__inference_Batch_Norm_5_256_layer_call_fn_1916624867

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_Batch_Norm_5_256_layer_call_and_return_conditional_losses_19166221962
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ì

P__inference_Pool_Concatenate_layer_call_and_return_conditional_losses_1916622776

inputs
inputs_1
inputs_2
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1inputs_2concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í
¥
1__inference_Conv1D_5_256_layer_call_fn_1916624723

inputs
unknown:
	unknown_0:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_Conv1D_5_256_layer_call_and_return_conditional_losses_19166227052
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

q
S__inference_Concatenate_Dropout_layer_call_and_return_conditional_losses_1916622783

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

K
/__inference_Max_Pool_3_layer_call_fn_1916622509

inputs
identityÔ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Max_Pool_3_layer_call_and_return_conditional_losses_19166225032
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

K
/__inference_Max_Pool_7_layer_call_fn_1916622535

inputs
identityÔ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Max_Pool_7_layer_call_and_return_conditional_losses_19166225292
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü
 
L__inference_Conv1D_7_256_layer_call_and_return_conditional_losses_1916622678

inputsC
+conv1d_expanddims_1_readvariableop_resource:.
biasadd_readvariableop_resource:	

identity_1¢BiasAdd/ReadVariableOp¢"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims/dim 
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
conv1d/ExpandDimsº
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim¹
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv1d/ExpandDims_1À
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAddo
SigmoidSigmoidBiasAdd:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
Sigmoidp
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
muli
IdentityIdentitymul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

IdentityÕ
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1916622671*V
_output_shapesD
B:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	IdentityN¶

Identity_1IdentityIdentityN:output:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª
p
Q__inference_Embedding_Dropout_layer_call_and_return_conditional_losses_1916624354

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Const
dropout/MulMulinputsdropout/Const:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeÂ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yÌ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1s
IdentityIdentitydropout/Mul_1:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
©
f
J__inference_Max_Pool_3_layer_call_and_return_conditional_losses_1916622503

inputs
identityp
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Max/reduction_indicest
MaxMaxinputsMax/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Maxi
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü*
í
P__inference_Batch_Norm_5_256_layer_call_and_return_conditional_losses_1916622256

inputs6
'assignmovingavg_readvariableop_resource:	8
)assignmovingavg_1_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	0
!batchnorm_readvariableop_resource:	
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:2
moments/StopGradient²
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices·
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay¥
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg/mul¿
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay«
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp¡
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ
³
P__inference_Batch_Norm_7_128_layer_call_and_return_conditional_losses_1916624650

inputs0
!batchnorm_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	2
#batchnorm_readvariableop_1_resource:	2
#batchnorm_readvariableop_2_resource:	
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1é
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü*
í
P__inference_Batch_Norm_3_128_layer_call_and_return_conditional_losses_1916621608

inputs6
'assignmovingavg_readvariableop_resource:	8
)assignmovingavg_1_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	0
!batchnorm_readvariableop_resource:	
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:2
moments/StopGradient²
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices·
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay¥
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg/mul¿
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay«
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp¡
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾
Ô
5__inference_Batch_Norm_7_128_layer_call_fn_1916624617

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_Batch_Norm_7_128_layer_call_and_return_conditional_losses_19166218722
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ö

P__inference_Pool_Concatenate_layer_call_and_return_conditional_losses_1916625029
inputs_0
inputs_1
inputs_2
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:R N
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/2
ïn
Û
L__inference_Tree_Encoder_layer_call_and_return_conditional_losses_1916623546

inputs/
conv1d_7_128_1916623451:&
conv1d_7_128_1916623453:	/
conv1d_5_128_1916623456:&
conv1d_5_128_1916623458:	/
conv1d_3_128_1916623461:&
conv1d_3_128_1916623463:	*
batch_norm_7_128_1916623466:	*
batch_norm_7_128_1916623468:	*
batch_norm_7_128_1916623470:	*
batch_norm_7_128_1916623472:	*
batch_norm_5_128_1916623475:	*
batch_norm_5_128_1916623477:	*
batch_norm_5_128_1916623479:	*
batch_norm_5_128_1916623481:	*
batch_norm_3_128_1916623484:	*
batch_norm_3_128_1916623486:	*
batch_norm_3_128_1916623488:	*
batch_norm_3_128_1916623490:	/
conv1d_7_256_1916623493:&
conv1d_7_256_1916623495:	/
conv1d_5_256_1916623498:&
conv1d_5_256_1916623500:	/
conv1d_3_256_1916623503:&
conv1d_3_256_1916623505:	*
batch_norm_7_256_1916623508:	*
batch_norm_7_256_1916623510:	*
batch_norm_7_256_1916623512:	*
batch_norm_7_256_1916623514:	*
batch_norm_5_256_1916623517:	*
batch_norm_5_256_1916623519:	*
batch_norm_5_256_1916623521:	*
batch_norm_5_256_1916623523:	*
batch_norm_3_256_1916623526:	*
batch_norm_3_256_1916623528:	*
batch_norm_3_256_1916623530:	*
batch_norm_3_256_1916623532:	(
prediction_1916623540:	#
prediction_1916623542:
identity¢(Batch_Norm_3_128/StatefulPartitionedCall¢(Batch_Norm_3_256/StatefulPartitionedCall¢(Batch_Norm_5_128/StatefulPartitionedCall¢(Batch_Norm_5_256/StatefulPartitionedCall¢(Batch_Norm_7_128/StatefulPartitionedCall¢(Batch_Norm_7_256/StatefulPartitionedCall¢+Concatenate_Dropout/StatefulPartitionedCall¢$Conv1D_3_128/StatefulPartitionedCall¢$Conv1D_3_256/StatefulPartitionedCall¢$Conv1D_5_128/StatefulPartitionedCall¢$Conv1D_5_256/StatefulPartitionedCall¢$Conv1D_7_128/StatefulPartitionedCall¢$Conv1D_7_256/StatefulPartitionedCall¢)Embedding_Dropout/StatefulPartitionedCall¢"Prediction/StatefulPartitionedCall
)Embedding_Dropout/StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_Embedding_Dropout_layer_call_and_return_conditional_losses_19166230032+
)Embedding_Dropout/StatefulPartitionedCallñ
$Conv1D_7_128/StatefulPartitionedCallStatefulPartitionedCall2Embedding_Dropout/StatefulPartitionedCall:output:0conv1d_7_128_1916623451conv1d_7_128_1916623453*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_Conv1D_7_128_layer_call_and_return_conditional_losses_19166225702&
$Conv1D_7_128/StatefulPartitionedCallñ
$Conv1D_5_128/StatefulPartitionedCallStatefulPartitionedCall2Embedding_Dropout/StatefulPartitionedCall:output:0conv1d_5_128_1916623456conv1d_5_128_1916623458*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_Conv1D_5_128_layer_call_and_return_conditional_losses_19166225972&
$Conv1D_5_128/StatefulPartitionedCallñ
$Conv1D_3_128/StatefulPartitionedCallStatefulPartitionedCall2Embedding_Dropout/StatefulPartitionedCall:output:0conv1d_3_128_1916623461conv1d_3_128_1916623463*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_Conv1D_3_128_layer_call_and_return_conditional_losses_19166226242&
$Conv1D_3_128/StatefulPartitionedCall¼
(Batch_Norm_7_128/StatefulPartitionedCallStatefulPartitionedCall-Conv1D_7_128/StatefulPartitionedCall:output:0batch_norm_7_128_1916623466batch_norm_7_128_1916623468batch_norm_7_128_1916623470batch_norm_7_128_1916623472*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_Batch_Norm_7_128_layer_call_and_return_conditional_losses_19166219322*
(Batch_Norm_7_128/StatefulPartitionedCall¼
(Batch_Norm_5_128/StatefulPartitionedCallStatefulPartitionedCall-Conv1D_5_128/StatefulPartitionedCall:output:0batch_norm_5_128_1916623475batch_norm_5_128_1916623477batch_norm_5_128_1916623479batch_norm_5_128_1916623481*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_Batch_Norm_5_128_layer_call_and_return_conditional_losses_19166217702*
(Batch_Norm_5_128/StatefulPartitionedCall¼
(Batch_Norm_3_128/StatefulPartitionedCallStatefulPartitionedCall-Conv1D_3_128/StatefulPartitionedCall:output:0batch_norm_3_128_1916623484batch_norm_3_128_1916623486batch_norm_3_128_1916623488batch_norm_3_128_1916623490*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_Batch_Norm_3_128_layer_call_and_return_conditional_losses_19166216082*
(Batch_Norm_3_128/StatefulPartitionedCallð
$Conv1D_7_256/StatefulPartitionedCallStatefulPartitionedCall1Batch_Norm_7_128/StatefulPartitionedCall:output:0conv1d_7_256_1916623493conv1d_7_256_1916623495*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_Conv1D_7_256_layer_call_and_return_conditional_losses_19166226782&
$Conv1D_7_256/StatefulPartitionedCallð
$Conv1D_5_256/StatefulPartitionedCallStatefulPartitionedCall1Batch_Norm_5_128/StatefulPartitionedCall:output:0conv1d_5_256_1916623498conv1d_5_256_1916623500*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_Conv1D_5_256_layer_call_and_return_conditional_losses_19166227052&
$Conv1D_5_256/StatefulPartitionedCallð
$Conv1D_3_256/StatefulPartitionedCallStatefulPartitionedCall1Batch_Norm_3_128/StatefulPartitionedCall:output:0conv1d_3_256_1916623503conv1d_3_256_1916623505*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_Conv1D_3_256_layer_call_and_return_conditional_losses_19166227322&
$Conv1D_3_256/StatefulPartitionedCall¼
(Batch_Norm_7_256/StatefulPartitionedCallStatefulPartitionedCall-Conv1D_7_256/StatefulPartitionedCall:output:0batch_norm_7_256_1916623508batch_norm_7_256_1916623510batch_norm_7_256_1916623512batch_norm_7_256_1916623514*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_Batch_Norm_7_256_layer_call_and_return_conditional_losses_19166224182*
(Batch_Norm_7_256/StatefulPartitionedCall¼
(Batch_Norm_5_256/StatefulPartitionedCallStatefulPartitionedCall-Conv1D_5_256/StatefulPartitionedCall:output:0batch_norm_5_256_1916623517batch_norm_5_256_1916623519batch_norm_5_256_1916623521batch_norm_5_256_1916623523*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_Batch_Norm_5_256_layer_call_and_return_conditional_losses_19166222562*
(Batch_Norm_5_256/StatefulPartitionedCall¼
(Batch_Norm_3_256/StatefulPartitionedCallStatefulPartitionedCall-Conv1D_3_256/StatefulPartitionedCall:output:0batch_norm_3_256_1916623526batch_norm_3_256_1916623528batch_norm_3_256_1916623530batch_norm_3_256_1916623532*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_Batch_Norm_3_256_layer_call_and_return_conditional_losses_19166220942*
(Batch_Norm_3_256/StatefulPartitionedCall
Max_Pool_3/PartitionedCallPartitionedCall1Batch_Norm_3_256/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Max_Pool_3_layer_call_and_return_conditional_losses_19166225032
Max_Pool_3/PartitionedCall
Max_Pool_5/PartitionedCallPartitionedCall1Batch_Norm_5_256/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Max_Pool_5_layer_call_and_return_conditional_losses_19166225162
Max_Pool_5/PartitionedCall
Max_Pool_7/PartitionedCallPartitionedCall1Batch_Norm_7_256/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Max_Pool_7_layer_call_and_return_conditional_losses_19166225292
Max_Pool_7/PartitionedCallÝ
 Pool_Concatenate/PartitionedCallPartitionedCall#Max_Pool_3/PartitionedCall:output:0#Max_Pool_5/PartitionedCall:output:0#Max_Pool_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_Pool_Concatenate_layer_call_and_return_conditional_losses_19166227762"
 Pool_Concatenate/PartitionedCallä
+Concatenate_Dropout/StatefulPartitionedCallStatefulPartitionedCall)Pool_Concatenate/PartitionedCall:output:0*^Embedding_Dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_Concatenate_Dropout_layer_call_and_return_conditional_losses_19166229122-
+Concatenate_Dropout/StatefulPartitionedCallÛ
"Prediction/StatefulPartitionedCallStatefulPartitionedCall4Concatenate_Dropout/StatefulPartitionedCall:output:0prediction_1916623540prediction_1916623542*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Prediction_layer_call_and_return_conditional_losses_19166227962$
"Prediction/StatefulPartitionedCallê
IdentityIdentity+Prediction/StatefulPartitionedCall:output:0)^Batch_Norm_3_128/StatefulPartitionedCall)^Batch_Norm_3_256/StatefulPartitionedCall)^Batch_Norm_5_128/StatefulPartitionedCall)^Batch_Norm_5_256/StatefulPartitionedCall)^Batch_Norm_7_128/StatefulPartitionedCall)^Batch_Norm_7_256/StatefulPartitionedCall,^Concatenate_Dropout/StatefulPartitionedCall%^Conv1D_3_128/StatefulPartitionedCall%^Conv1D_3_256/StatefulPartitionedCall%^Conv1D_5_128/StatefulPartitionedCall%^Conv1D_5_256/StatefulPartitionedCall%^Conv1D_7_128/StatefulPartitionedCall%^Conv1D_7_256/StatefulPartitionedCall*^Embedding_Dropout/StatefulPartitionedCall#^Prediction/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapeso
m:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2T
(Batch_Norm_3_128/StatefulPartitionedCall(Batch_Norm_3_128/StatefulPartitionedCall2T
(Batch_Norm_3_256/StatefulPartitionedCall(Batch_Norm_3_256/StatefulPartitionedCall2T
(Batch_Norm_5_128/StatefulPartitionedCall(Batch_Norm_5_128/StatefulPartitionedCall2T
(Batch_Norm_5_256/StatefulPartitionedCall(Batch_Norm_5_256/StatefulPartitionedCall2T
(Batch_Norm_7_128/StatefulPartitionedCall(Batch_Norm_7_128/StatefulPartitionedCall2T
(Batch_Norm_7_256/StatefulPartitionedCall(Batch_Norm_7_256/StatefulPartitionedCall2Z
+Concatenate_Dropout/StatefulPartitionedCall+Concatenate_Dropout/StatefulPartitionedCall2L
$Conv1D_3_128/StatefulPartitionedCall$Conv1D_3_128/StatefulPartitionedCall2L
$Conv1D_3_256/StatefulPartitionedCall$Conv1D_3_256/StatefulPartitionedCall2L
$Conv1D_5_128/StatefulPartitionedCall$Conv1D_5_128/StatefulPartitionedCall2L
$Conv1D_5_256/StatefulPartitionedCall$Conv1D_5_256/StatefulPartitionedCall2L
$Conv1D_7_128/StatefulPartitionedCall$Conv1D_7_128/StatefulPartitionedCall2L
$Conv1D_7_256/StatefulPartitionedCall$Conv1D_7_256/StatefulPartitionedCall2V
)Embedding_Dropout/StatefulPartitionedCall)Embedding_Dropout/StatefulPartitionedCall2H
"Prediction/StatefulPartitionedCall"Prediction/StatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameInputs
£È
Ü"
L__inference_Tree_Encoder_layer_call_and_return_conditional_losses_1916624010

inputsP
8conv1d_7_128_conv1d_expanddims_1_readvariableop_resource:;
,conv1d_7_128_biasadd_readvariableop_resource:	P
8conv1d_5_128_conv1d_expanddims_1_readvariableop_resource:;
,conv1d_5_128_biasadd_readvariableop_resource:	P
8conv1d_3_128_conv1d_expanddims_1_readvariableop_resource:;
,conv1d_3_128_biasadd_readvariableop_resource:	A
2batch_norm_7_128_batchnorm_readvariableop_resource:	E
6batch_norm_7_128_batchnorm_mul_readvariableop_resource:	C
4batch_norm_7_128_batchnorm_readvariableop_1_resource:	C
4batch_norm_7_128_batchnorm_readvariableop_2_resource:	A
2batch_norm_5_128_batchnorm_readvariableop_resource:	E
6batch_norm_5_128_batchnorm_mul_readvariableop_resource:	C
4batch_norm_5_128_batchnorm_readvariableop_1_resource:	C
4batch_norm_5_128_batchnorm_readvariableop_2_resource:	A
2batch_norm_3_128_batchnorm_readvariableop_resource:	E
6batch_norm_3_128_batchnorm_mul_readvariableop_resource:	C
4batch_norm_3_128_batchnorm_readvariableop_1_resource:	C
4batch_norm_3_128_batchnorm_readvariableop_2_resource:	P
8conv1d_7_256_conv1d_expanddims_1_readvariableop_resource:;
,conv1d_7_256_biasadd_readvariableop_resource:	P
8conv1d_5_256_conv1d_expanddims_1_readvariableop_resource:;
,conv1d_5_256_biasadd_readvariableop_resource:	P
8conv1d_3_256_conv1d_expanddims_1_readvariableop_resource:;
,conv1d_3_256_biasadd_readvariableop_resource:	A
2batch_norm_7_256_batchnorm_readvariableop_resource:	E
6batch_norm_7_256_batchnorm_mul_readvariableop_resource:	C
4batch_norm_7_256_batchnorm_readvariableop_1_resource:	C
4batch_norm_7_256_batchnorm_readvariableop_2_resource:	A
2batch_norm_5_256_batchnorm_readvariableop_resource:	E
6batch_norm_5_256_batchnorm_mul_readvariableop_resource:	C
4batch_norm_5_256_batchnorm_readvariableop_1_resource:	C
4batch_norm_5_256_batchnorm_readvariableop_2_resource:	A
2batch_norm_3_256_batchnorm_readvariableop_resource:	E
6batch_norm_3_256_batchnorm_mul_readvariableop_resource:	C
4batch_norm_3_256_batchnorm_readvariableop_1_resource:	C
4batch_norm_3_256_batchnorm_readvariableop_2_resource:	<
)prediction_matmul_readvariableop_resource:	8
*prediction_biasadd_readvariableop_resource:
identity¢)Batch_Norm_3_128/batchnorm/ReadVariableOp¢+Batch_Norm_3_128/batchnorm/ReadVariableOp_1¢+Batch_Norm_3_128/batchnorm/ReadVariableOp_2¢-Batch_Norm_3_128/batchnorm/mul/ReadVariableOp¢)Batch_Norm_3_256/batchnorm/ReadVariableOp¢+Batch_Norm_3_256/batchnorm/ReadVariableOp_1¢+Batch_Norm_3_256/batchnorm/ReadVariableOp_2¢-Batch_Norm_3_256/batchnorm/mul/ReadVariableOp¢)Batch_Norm_5_128/batchnorm/ReadVariableOp¢+Batch_Norm_5_128/batchnorm/ReadVariableOp_1¢+Batch_Norm_5_128/batchnorm/ReadVariableOp_2¢-Batch_Norm_5_128/batchnorm/mul/ReadVariableOp¢)Batch_Norm_5_256/batchnorm/ReadVariableOp¢+Batch_Norm_5_256/batchnorm/ReadVariableOp_1¢+Batch_Norm_5_256/batchnorm/ReadVariableOp_2¢-Batch_Norm_5_256/batchnorm/mul/ReadVariableOp¢)Batch_Norm_7_128/batchnorm/ReadVariableOp¢+Batch_Norm_7_128/batchnorm/ReadVariableOp_1¢+Batch_Norm_7_128/batchnorm/ReadVariableOp_2¢-Batch_Norm_7_128/batchnorm/mul/ReadVariableOp¢)Batch_Norm_7_256/batchnorm/ReadVariableOp¢+Batch_Norm_7_256/batchnorm/ReadVariableOp_1¢+Batch_Norm_7_256/batchnorm/ReadVariableOp_2¢-Batch_Norm_7_256/batchnorm/mul/ReadVariableOp¢#Conv1D_3_128/BiasAdd/ReadVariableOp¢/Conv1D_3_128/conv1d/ExpandDims_1/ReadVariableOp¢#Conv1D_3_256/BiasAdd/ReadVariableOp¢/Conv1D_3_256/conv1d/ExpandDims_1/ReadVariableOp¢#Conv1D_5_128/BiasAdd/ReadVariableOp¢/Conv1D_5_128/conv1d/ExpandDims_1/ReadVariableOp¢#Conv1D_5_256/BiasAdd/ReadVariableOp¢/Conv1D_5_256/conv1d/ExpandDims_1/ReadVariableOp¢#Conv1D_7_128/BiasAdd/ReadVariableOp¢/Conv1D_7_128/conv1d/ExpandDims_1/ReadVariableOp¢#Conv1D_7_256/BiasAdd/ReadVariableOp¢/Conv1D_7_256/conv1d/ExpandDims_1/ReadVariableOp¢!Prediction/BiasAdd/ReadVariableOp¢ Prediction/MatMul/ReadVariableOp
Embedding_Dropout/IdentityIdentityinputs*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Embedding_Dropout/Identity
"Conv1D_7_128/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2$
"Conv1D_7_128/conv1d/ExpandDims/dimä
Conv1D_7_128/conv1d/ExpandDims
ExpandDims#Embedding_Dropout/Identity:output:0+Conv1D_7_128/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2 
Conv1D_7_128/conv1d/ExpandDimsá
/Conv1D_7_128/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp8conv1d_7_128_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype021
/Conv1D_7_128/conv1d/ExpandDims_1/ReadVariableOp
$Conv1D_7_128/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Conv1D_7_128/conv1d/ExpandDims_1/dimí
 Conv1D_7_128/conv1d/ExpandDims_1
ExpandDims7Conv1D_7_128/conv1d/ExpandDims_1/ReadVariableOp:value:0-Conv1D_7_128/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2"
 Conv1D_7_128/conv1d/ExpandDims_1ô
Conv1D_7_128/conv1dConv2D'Conv1D_7_128/conv1d/ExpandDims:output:0)Conv1D_7_128/conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv1D_7_128/conv1dÃ
Conv1D_7_128/conv1d/SqueezeSqueezeConv1D_7_128/conv1d:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
Conv1D_7_128/conv1d/Squeeze´
#Conv1D_7_128/BiasAdd/ReadVariableOpReadVariableOp,conv1d_7_128_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#Conv1D_7_128/BiasAdd/ReadVariableOpÊ
Conv1D_7_128/BiasAddBiasAdd$Conv1D_7_128/conv1d/Squeeze:output:0+Conv1D_7_128/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_7_128/BiasAdd
Conv1D_7_128/SigmoidSigmoidConv1D_7_128/BiasAdd:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_7_128/Sigmoid¤
Conv1D_7_128/mulMulConv1D_7_128/BiasAdd:output:0Conv1D_7_128/Sigmoid:y:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_7_128/mul
Conv1D_7_128/IdentityIdentityConv1D_7_128/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_7_128/Identity
Conv1D_7_128/IdentityN	IdentityNConv1D_7_128/mul:z:0Conv1D_7_128/BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1916623806*V
_output_shapesD
B:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_7_128/IdentityN
"Conv1D_5_128/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2$
"Conv1D_5_128/conv1d/ExpandDims/dimä
Conv1D_5_128/conv1d/ExpandDims
ExpandDims#Embedding_Dropout/Identity:output:0+Conv1D_5_128/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2 
Conv1D_5_128/conv1d/ExpandDimsá
/Conv1D_5_128/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp8conv1d_5_128_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype021
/Conv1D_5_128/conv1d/ExpandDims_1/ReadVariableOp
$Conv1D_5_128/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Conv1D_5_128/conv1d/ExpandDims_1/dimí
 Conv1D_5_128/conv1d/ExpandDims_1
ExpandDims7Conv1D_5_128/conv1d/ExpandDims_1/ReadVariableOp:value:0-Conv1D_5_128/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2"
 Conv1D_5_128/conv1d/ExpandDims_1ô
Conv1D_5_128/conv1dConv2D'Conv1D_5_128/conv1d/ExpandDims:output:0)Conv1D_5_128/conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv1D_5_128/conv1dÃ
Conv1D_5_128/conv1d/SqueezeSqueezeConv1D_5_128/conv1d:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
Conv1D_5_128/conv1d/Squeeze´
#Conv1D_5_128/BiasAdd/ReadVariableOpReadVariableOp,conv1d_5_128_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#Conv1D_5_128/BiasAdd/ReadVariableOpÊ
Conv1D_5_128/BiasAddBiasAdd$Conv1D_5_128/conv1d/Squeeze:output:0+Conv1D_5_128/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_5_128/BiasAdd
Conv1D_5_128/SigmoidSigmoidConv1D_5_128/BiasAdd:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_5_128/Sigmoid¤
Conv1D_5_128/mulMulConv1D_5_128/BiasAdd:output:0Conv1D_5_128/Sigmoid:y:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_5_128/mul
Conv1D_5_128/IdentityIdentityConv1D_5_128/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_5_128/Identity
Conv1D_5_128/IdentityN	IdentityNConv1D_5_128/mul:z:0Conv1D_5_128/BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1916623823*V
_output_shapesD
B:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_5_128/IdentityN
"Conv1D_3_128/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2$
"Conv1D_3_128/conv1d/ExpandDims/dimä
Conv1D_3_128/conv1d/ExpandDims
ExpandDims#Embedding_Dropout/Identity:output:0+Conv1D_3_128/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2 
Conv1D_3_128/conv1d/ExpandDimsá
/Conv1D_3_128/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp8conv1d_3_128_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype021
/Conv1D_3_128/conv1d/ExpandDims_1/ReadVariableOp
$Conv1D_3_128/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Conv1D_3_128/conv1d/ExpandDims_1/dimí
 Conv1D_3_128/conv1d/ExpandDims_1
ExpandDims7Conv1D_3_128/conv1d/ExpandDims_1/ReadVariableOp:value:0-Conv1D_3_128/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2"
 Conv1D_3_128/conv1d/ExpandDims_1ô
Conv1D_3_128/conv1dConv2D'Conv1D_3_128/conv1d/ExpandDims:output:0)Conv1D_3_128/conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv1D_3_128/conv1dÃ
Conv1D_3_128/conv1d/SqueezeSqueezeConv1D_3_128/conv1d:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
Conv1D_3_128/conv1d/Squeeze´
#Conv1D_3_128/BiasAdd/ReadVariableOpReadVariableOp,conv1d_3_128_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#Conv1D_3_128/BiasAdd/ReadVariableOpÊ
Conv1D_3_128/BiasAddBiasAdd$Conv1D_3_128/conv1d/Squeeze:output:0+Conv1D_3_128/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_3_128/BiasAdd
Conv1D_3_128/SigmoidSigmoidConv1D_3_128/BiasAdd:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_3_128/Sigmoid¤
Conv1D_3_128/mulMulConv1D_3_128/BiasAdd:output:0Conv1D_3_128/Sigmoid:y:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_3_128/mul
Conv1D_3_128/IdentityIdentityConv1D_3_128/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_3_128/Identity
Conv1D_3_128/IdentityN	IdentityNConv1D_3_128/mul:z:0Conv1D_3_128/BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1916623840*V
_output_shapesD
B:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_3_128/IdentityNÆ
)Batch_Norm_7_128/batchnorm/ReadVariableOpReadVariableOp2batch_norm_7_128_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02+
)Batch_Norm_7_128/batchnorm/ReadVariableOp
 Batch_Norm_7_128/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2"
 Batch_Norm_7_128/batchnorm/add/yÍ
Batch_Norm_7_128/batchnorm/addAddV21Batch_Norm_7_128/batchnorm/ReadVariableOp:value:0)Batch_Norm_7_128/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2 
Batch_Norm_7_128/batchnorm/add
 Batch_Norm_7_128/batchnorm/RsqrtRsqrt"Batch_Norm_7_128/batchnorm/add:z:0*
T0*
_output_shapes	
:2"
 Batch_Norm_7_128/batchnorm/RsqrtÒ
-Batch_Norm_7_128/batchnorm/mul/ReadVariableOpReadVariableOp6batch_norm_7_128_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02/
-Batch_Norm_7_128/batchnorm/mul/ReadVariableOpÊ
Batch_Norm_7_128/batchnorm/mulMul$Batch_Norm_7_128/batchnorm/Rsqrt:y:05Batch_Norm_7_128/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2 
Batch_Norm_7_128/batchnorm/mulÐ
 Batch_Norm_7_128/batchnorm/mul_1MulConv1D_7_128/IdentityN:output:0"Batch_Norm_7_128/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2"
 Batch_Norm_7_128/batchnorm/mul_1Ì
+Batch_Norm_7_128/batchnorm/ReadVariableOp_1ReadVariableOp4batch_norm_7_128_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02-
+Batch_Norm_7_128/batchnorm/ReadVariableOp_1Ê
 Batch_Norm_7_128/batchnorm/mul_2Mul3Batch_Norm_7_128/batchnorm/ReadVariableOp_1:value:0"Batch_Norm_7_128/batchnorm/mul:z:0*
T0*
_output_shapes	
:2"
 Batch_Norm_7_128/batchnorm/mul_2Ì
+Batch_Norm_7_128/batchnorm/ReadVariableOp_2ReadVariableOp4batch_norm_7_128_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02-
+Batch_Norm_7_128/batchnorm/ReadVariableOp_2È
Batch_Norm_7_128/batchnorm/subSub3Batch_Norm_7_128/batchnorm/ReadVariableOp_2:value:0$Batch_Norm_7_128/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2 
Batch_Norm_7_128/batchnorm/sub×
 Batch_Norm_7_128/batchnorm/add_1AddV2$Batch_Norm_7_128/batchnorm/mul_1:z:0"Batch_Norm_7_128/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2"
 Batch_Norm_7_128/batchnorm/add_1Æ
)Batch_Norm_5_128/batchnorm/ReadVariableOpReadVariableOp2batch_norm_5_128_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02+
)Batch_Norm_5_128/batchnorm/ReadVariableOp
 Batch_Norm_5_128/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2"
 Batch_Norm_5_128/batchnorm/add/yÍ
Batch_Norm_5_128/batchnorm/addAddV21Batch_Norm_5_128/batchnorm/ReadVariableOp:value:0)Batch_Norm_5_128/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2 
Batch_Norm_5_128/batchnorm/add
 Batch_Norm_5_128/batchnorm/RsqrtRsqrt"Batch_Norm_5_128/batchnorm/add:z:0*
T0*
_output_shapes	
:2"
 Batch_Norm_5_128/batchnorm/RsqrtÒ
-Batch_Norm_5_128/batchnorm/mul/ReadVariableOpReadVariableOp6batch_norm_5_128_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02/
-Batch_Norm_5_128/batchnorm/mul/ReadVariableOpÊ
Batch_Norm_5_128/batchnorm/mulMul$Batch_Norm_5_128/batchnorm/Rsqrt:y:05Batch_Norm_5_128/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2 
Batch_Norm_5_128/batchnorm/mulÐ
 Batch_Norm_5_128/batchnorm/mul_1MulConv1D_5_128/IdentityN:output:0"Batch_Norm_5_128/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2"
 Batch_Norm_5_128/batchnorm/mul_1Ì
+Batch_Norm_5_128/batchnorm/ReadVariableOp_1ReadVariableOp4batch_norm_5_128_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02-
+Batch_Norm_5_128/batchnorm/ReadVariableOp_1Ê
 Batch_Norm_5_128/batchnorm/mul_2Mul3Batch_Norm_5_128/batchnorm/ReadVariableOp_1:value:0"Batch_Norm_5_128/batchnorm/mul:z:0*
T0*
_output_shapes	
:2"
 Batch_Norm_5_128/batchnorm/mul_2Ì
+Batch_Norm_5_128/batchnorm/ReadVariableOp_2ReadVariableOp4batch_norm_5_128_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02-
+Batch_Norm_5_128/batchnorm/ReadVariableOp_2È
Batch_Norm_5_128/batchnorm/subSub3Batch_Norm_5_128/batchnorm/ReadVariableOp_2:value:0$Batch_Norm_5_128/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2 
Batch_Norm_5_128/batchnorm/sub×
 Batch_Norm_5_128/batchnorm/add_1AddV2$Batch_Norm_5_128/batchnorm/mul_1:z:0"Batch_Norm_5_128/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2"
 Batch_Norm_5_128/batchnorm/add_1Æ
)Batch_Norm_3_128/batchnorm/ReadVariableOpReadVariableOp2batch_norm_3_128_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02+
)Batch_Norm_3_128/batchnorm/ReadVariableOp
 Batch_Norm_3_128/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2"
 Batch_Norm_3_128/batchnorm/add/yÍ
Batch_Norm_3_128/batchnorm/addAddV21Batch_Norm_3_128/batchnorm/ReadVariableOp:value:0)Batch_Norm_3_128/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2 
Batch_Norm_3_128/batchnorm/add
 Batch_Norm_3_128/batchnorm/RsqrtRsqrt"Batch_Norm_3_128/batchnorm/add:z:0*
T0*
_output_shapes	
:2"
 Batch_Norm_3_128/batchnorm/RsqrtÒ
-Batch_Norm_3_128/batchnorm/mul/ReadVariableOpReadVariableOp6batch_norm_3_128_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02/
-Batch_Norm_3_128/batchnorm/mul/ReadVariableOpÊ
Batch_Norm_3_128/batchnorm/mulMul$Batch_Norm_3_128/batchnorm/Rsqrt:y:05Batch_Norm_3_128/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2 
Batch_Norm_3_128/batchnorm/mulÐ
 Batch_Norm_3_128/batchnorm/mul_1MulConv1D_3_128/IdentityN:output:0"Batch_Norm_3_128/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2"
 Batch_Norm_3_128/batchnorm/mul_1Ì
+Batch_Norm_3_128/batchnorm/ReadVariableOp_1ReadVariableOp4batch_norm_3_128_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02-
+Batch_Norm_3_128/batchnorm/ReadVariableOp_1Ê
 Batch_Norm_3_128/batchnorm/mul_2Mul3Batch_Norm_3_128/batchnorm/ReadVariableOp_1:value:0"Batch_Norm_3_128/batchnorm/mul:z:0*
T0*
_output_shapes	
:2"
 Batch_Norm_3_128/batchnorm/mul_2Ì
+Batch_Norm_3_128/batchnorm/ReadVariableOp_2ReadVariableOp4batch_norm_3_128_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02-
+Batch_Norm_3_128/batchnorm/ReadVariableOp_2È
Batch_Norm_3_128/batchnorm/subSub3Batch_Norm_3_128/batchnorm/ReadVariableOp_2:value:0$Batch_Norm_3_128/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2 
Batch_Norm_3_128/batchnorm/sub×
 Batch_Norm_3_128/batchnorm/add_1AddV2$Batch_Norm_3_128/batchnorm/mul_1:z:0"Batch_Norm_3_128/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2"
 Batch_Norm_3_128/batchnorm/add_1
"Conv1D_7_256/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2$
"Conv1D_7_256/conv1d/ExpandDims/dimå
Conv1D_7_256/conv1d/ExpandDims
ExpandDims$Batch_Norm_7_128/batchnorm/add_1:z:0+Conv1D_7_256/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2 
Conv1D_7_256/conv1d/ExpandDimsá
/Conv1D_7_256/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp8conv1d_7_256_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype021
/Conv1D_7_256/conv1d/ExpandDims_1/ReadVariableOp
$Conv1D_7_256/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Conv1D_7_256/conv1d/ExpandDims_1/dimí
 Conv1D_7_256/conv1d/ExpandDims_1
ExpandDims7Conv1D_7_256/conv1d/ExpandDims_1/ReadVariableOp:value:0-Conv1D_7_256/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2"
 Conv1D_7_256/conv1d/ExpandDims_1ô
Conv1D_7_256/conv1dConv2D'Conv1D_7_256/conv1d/ExpandDims:output:0)Conv1D_7_256/conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv1D_7_256/conv1dÃ
Conv1D_7_256/conv1d/SqueezeSqueezeConv1D_7_256/conv1d:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
Conv1D_7_256/conv1d/Squeeze´
#Conv1D_7_256/BiasAdd/ReadVariableOpReadVariableOp,conv1d_7_256_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#Conv1D_7_256/BiasAdd/ReadVariableOpÊ
Conv1D_7_256/BiasAddBiasAdd$Conv1D_7_256/conv1d/Squeeze:output:0+Conv1D_7_256/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_7_256/BiasAdd
Conv1D_7_256/SigmoidSigmoidConv1D_7_256/BiasAdd:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_7_256/Sigmoid¤
Conv1D_7_256/mulMulConv1D_7_256/BiasAdd:output:0Conv1D_7_256/Sigmoid:y:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_7_256/mul
Conv1D_7_256/IdentityIdentityConv1D_7_256/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_7_256/Identity
Conv1D_7_256/IdentityN	IdentityNConv1D_7_256/mul:z:0Conv1D_7_256/BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1916623905*V
_output_shapesD
B:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_7_256/IdentityN
"Conv1D_5_256/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2$
"Conv1D_5_256/conv1d/ExpandDims/dimå
Conv1D_5_256/conv1d/ExpandDims
ExpandDims$Batch_Norm_5_128/batchnorm/add_1:z:0+Conv1D_5_256/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2 
Conv1D_5_256/conv1d/ExpandDimsá
/Conv1D_5_256/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp8conv1d_5_256_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype021
/Conv1D_5_256/conv1d/ExpandDims_1/ReadVariableOp
$Conv1D_5_256/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Conv1D_5_256/conv1d/ExpandDims_1/dimí
 Conv1D_5_256/conv1d/ExpandDims_1
ExpandDims7Conv1D_5_256/conv1d/ExpandDims_1/ReadVariableOp:value:0-Conv1D_5_256/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2"
 Conv1D_5_256/conv1d/ExpandDims_1ô
Conv1D_5_256/conv1dConv2D'Conv1D_5_256/conv1d/ExpandDims:output:0)Conv1D_5_256/conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv1D_5_256/conv1dÃ
Conv1D_5_256/conv1d/SqueezeSqueezeConv1D_5_256/conv1d:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
Conv1D_5_256/conv1d/Squeeze´
#Conv1D_5_256/BiasAdd/ReadVariableOpReadVariableOp,conv1d_5_256_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#Conv1D_5_256/BiasAdd/ReadVariableOpÊ
Conv1D_5_256/BiasAddBiasAdd$Conv1D_5_256/conv1d/Squeeze:output:0+Conv1D_5_256/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_5_256/BiasAdd
Conv1D_5_256/SigmoidSigmoidConv1D_5_256/BiasAdd:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_5_256/Sigmoid¤
Conv1D_5_256/mulMulConv1D_5_256/BiasAdd:output:0Conv1D_5_256/Sigmoid:y:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_5_256/mul
Conv1D_5_256/IdentityIdentityConv1D_5_256/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_5_256/Identity
Conv1D_5_256/IdentityN	IdentityNConv1D_5_256/mul:z:0Conv1D_5_256/BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1916623922*V
_output_shapesD
B:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_5_256/IdentityN
"Conv1D_3_256/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2$
"Conv1D_3_256/conv1d/ExpandDims/dimå
Conv1D_3_256/conv1d/ExpandDims
ExpandDims$Batch_Norm_3_128/batchnorm/add_1:z:0+Conv1D_3_256/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2 
Conv1D_3_256/conv1d/ExpandDimsá
/Conv1D_3_256/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp8conv1d_3_256_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype021
/Conv1D_3_256/conv1d/ExpandDims_1/ReadVariableOp
$Conv1D_3_256/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Conv1D_3_256/conv1d/ExpandDims_1/dimí
 Conv1D_3_256/conv1d/ExpandDims_1
ExpandDims7Conv1D_3_256/conv1d/ExpandDims_1/ReadVariableOp:value:0-Conv1D_3_256/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2"
 Conv1D_3_256/conv1d/ExpandDims_1ô
Conv1D_3_256/conv1dConv2D'Conv1D_3_256/conv1d/ExpandDims:output:0)Conv1D_3_256/conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv1D_3_256/conv1dÃ
Conv1D_3_256/conv1d/SqueezeSqueezeConv1D_3_256/conv1d:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
Conv1D_3_256/conv1d/Squeeze´
#Conv1D_3_256/BiasAdd/ReadVariableOpReadVariableOp,conv1d_3_256_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#Conv1D_3_256/BiasAdd/ReadVariableOpÊ
Conv1D_3_256/BiasAddBiasAdd$Conv1D_3_256/conv1d/Squeeze:output:0+Conv1D_3_256/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_3_256/BiasAdd
Conv1D_3_256/SigmoidSigmoidConv1D_3_256/BiasAdd:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_3_256/Sigmoid¤
Conv1D_3_256/mulMulConv1D_3_256/BiasAdd:output:0Conv1D_3_256/Sigmoid:y:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_3_256/mul
Conv1D_3_256/IdentityIdentityConv1D_3_256/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_3_256/Identity
Conv1D_3_256/IdentityN	IdentityNConv1D_3_256/mul:z:0Conv1D_3_256/BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1916623939*V
_output_shapesD
B:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_3_256/IdentityNÆ
)Batch_Norm_7_256/batchnorm/ReadVariableOpReadVariableOp2batch_norm_7_256_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02+
)Batch_Norm_7_256/batchnorm/ReadVariableOp
 Batch_Norm_7_256/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2"
 Batch_Norm_7_256/batchnorm/add/yÍ
Batch_Norm_7_256/batchnorm/addAddV21Batch_Norm_7_256/batchnorm/ReadVariableOp:value:0)Batch_Norm_7_256/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2 
Batch_Norm_7_256/batchnorm/add
 Batch_Norm_7_256/batchnorm/RsqrtRsqrt"Batch_Norm_7_256/batchnorm/add:z:0*
T0*
_output_shapes	
:2"
 Batch_Norm_7_256/batchnorm/RsqrtÒ
-Batch_Norm_7_256/batchnorm/mul/ReadVariableOpReadVariableOp6batch_norm_7_256_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02/
-Batch_Norm_7_256/batchnorm/mul/ReadVariableOpÊ
Batch_Norm_7_256/batchnorm/mulMul$Batch_Norm_7_256/batchnorm/Rsqrt:y:05Batch_Norm_7_256/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2 
Batch_Norm_7_256/batchnorm/mulÐ
 Batch_Norm_7_256/batchnorm/mul_1MulConv1D_7_256/IdentityN:output:0"Batch_Norm_7_256/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2"
 Batch_Norm_7_256/batchnorm/mul_1Ì
+Batch_Norm_7_256/batchnorm/ReadVariableOp_1ReadVariableOp4batch_norm_7_256_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02-
+Batch_Norm_7_256/batchnorm/ReadVariableOp_1Ê
 Batch_Norm_7_256/batchnorm/mul_2Mul3Batch_Norm_7_256/batchnorm/ReadVariableOp_1:value:0"Batch_Norm_7_256/batchnorm/mul:z:0*
T0*
_output_shapes	
:2"
 Batch_Norm_7_256/batchnorm/mul_2Ì
+Batch_Norm_7_256/batchnorm/ReadVariableOp_2ReadVariableOp4batch_norm_7_256_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02-
+Batch_Norm_7_256/batchnorm/ReadVariableOp_2È
Batch_Norm_7_256/batchnorm/subSub3Batch_Norm_7_256/batchnorm/ReadVariableOp_2:value:0$Batch_Norm_7_256/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2 
Batch_Norm_7_256/batchnorm/sub×
 Batch_Norm_7_256/batchnorm/add_1AddV2$Batch_Norm_7_256/batchnorm/mul_1:z:0"Batch_Norm_7_256/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2"
 Batch_Norm_7_256/batchnorm/add_1Æ
)Batch_Norm_5_256/batchnorm/ReadVariableOpReadVariableOp2batch_norm_5_256_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02+
)Batch_Norm_5_256/batchnorm/ReadVariableOp
 Batch_Norm_5_256/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2"
 Batch_Norm_5_256/batchnorm/add/yÍ
Batch_Norm_5_256/batchnorm/addAddV21Batch_Norm_5_256/batchnorm/ReadVariableOp:value:0)Batch_Norm_5_256/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2 
Batch_Norm_5_256/batchnorm/add
 Batch_Norm_5_256/batchnorm/RsqrtRsqrt"Batch_Norm_5_256/batchnorm/add:z:0*
T0*
_output_shapes	
:2"
 Batch_Norm_5_256/batchnorm/RsqrtÒ
-Batch_Norm_5_256/batchnorm/mul/ReadVariableOpReadVariableOp6batch_norm_5_256_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02/
-Batch_Norm_5_256/batchnorm/mul/ReadVariableOpÊ
Batch_Norm_5_256/batchnorm/mulMul$Batch_Norm_5_256/batchnorm/Rsqrt:y:05Batch_Norm_5_256/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2 
Batch_Norm_5_256/batchnorm/mulÐ
 Batch_Norm_5_256/batchnorm/mul_1MulConv1D_5_256/IdentityN:output:0"Batch_Norm_5_256/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2"
 Batch_Norm_5_256/batchnorm/mul_1Ì
+Batch_Norm_5_256/batchnorm/ReadVariableOp_1ReadVariableOp4batch_norm_5_256_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02-
+Batch_Norm_5_256/batchnorm/ReadVariableOp_1Ê
 Batch_Norm_5_256/batchnorm/mul_2Mul3Batch_Norm_5_256/batchnorm/ReadVariableOp_1:value:0"Batch_Norm_5_256/batchnorm/mul:z:0*
T0*
_output_shapes	
:2"
 Batch_Norm_5_256/batchnorm/mul_2Ì
+Batch_Norm_5_256/batchnorm/ReadVariableOp_2ReadVariableOp4batch_norm_5_256_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02-
+Batch_Norm_5_256/batchnorm/ReadVariableOp_2È
Batch_Norm_5_256/batchnorm/subSub3Batch_Norm_5_256/batchnorm/ReadVariableOp_2:value:0$Batch_Norm_5_256/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2 
Batch_Norm_5_256/batchnorm/sub×
 Batch_Norm_5_256/batchnorm/add_1AddV2$Batch_Norm_5_256/batchnorm/mul_1:z:0"Batch_Norm_5_256/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2"
 Batch_Norm_5_256/batchnorm/add_1Æ
)Batch_Norm_3_256/batchnorm/ReadVariableOpReadVariableOp2batch_norm_3_256_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02+
)Batch_Norm_3_256/batchnorm/ReadVariableOp
 Batch_Norm_3_256/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2"
 Batch_Norm_3_256/batchnorm/add/yÍ
Batch_Norm_3_256/batchnorm/addAddV21Batch_Norm_3_256/batchnorm/ReadVariableOp:value:0)Batch_Norm_3_256/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2 
Batch_Norm_3_256/batchnorm/add
 Batch_Norm_3_256/batchnorm/RsqrtRsqrt"Batch_Norm_3_256/batchnorm/add:z:0*
T0*
_output_shapes	
:2"
 Batch_Norm_3_256/batchnorm/RsqrtÒ
-Batch_Norm_3_256/batchnorm/mul/ReadVariableOpReadVariableOp6batch_norm_3_256_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02/
-Batch_Norm_3_256/batchnorm/mul/ReadVariableOpÊ
Batch_Norm_3_256/batchnorm/mulMul$Batch_Norm_3_256/batchnorm/Rsqrt:y:05Batch_Norm_3_256/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2 
Batch_Norm_3_256/batchnorm/mulÐ
 Batch_Norm_3_256/batchnorm/mul_1MulConv1D_3_256/IdentityN:output:0"Batch_Norm_3_256/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2"
 Batch_Norm_3_256/batchnorm/mul_1Ì
+Batch_Norm_3_256/batchnorm/ReadVariableOp_1ReadVariableOp4batch_norm_3_256_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02-
+Batch_Norm_3_256/batchnorm/ReadVariableOp_1Ê
 Batch_Norm_3_256/batchnorm/mul_2Mul3Batch_Norm_3_256/batchnorm/ReadVariableOp_1:value:0"Batch_Norm_3_256/batchnorm/mul:z:0*
T0*
_output_shapes	
:2"
 Batch_Norm_3_256/batchnorm/mul_2Ì
+Batch_Norm_3_256/batchnorm/ReadVariableOp_2ReadVariableOp4batch_norm_3_256_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02-
+Batch_Norm_3_256/batchnorm/ReadVariableOp_2È
Batch_Norm_3_256/batchnorm/subSub3Batch_Norm_3_256/batchnorm/ReadVariableOp_2:value:0$Batch_Norm_3_256/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2 
Batch_Norm_3_256/batchnorm/sub×
 Batch_Norm_3_256/batchnorm/add_1AddV2$Batch_Norm_3_256/batchnorm/mul_1:z:0"Batch_Norm_3_256/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2"
 Batch_Norm_3_256/batchnorm/add_1
 Max_Pool_3/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2"
 Max_Pool_3/Max/reduction_indices«
Max_Pool_3/MaxMax$Batch_Norm_3_256/batchnorm/add_1:z:0)Max_Pool_3/Max/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Max_Pool_3/Max
 Max_Pool_5/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2"
 Max_Pool_5/Max/reduction_indices«
Max_Pool_5/MaxMax$Batch_Norm_5_256/batchnorm/add_1:z:0)Max_Pool_5/Max/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Max_Pool_5/Max
 Max_Pool_7/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2"
 Max_Pool_7/Max/reduction_indices«
Max_Pool_7/MaxMax$Batch_Norm_7_256/batchnorm/add_1:z:0)Max_Pool_7/Max/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Max_Pool_7/Max~
Pool_Concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
Pool_Concatenate/concat/axisì
Pool_Concatenate/concatConcatV2Max_Pool_3/Max:output:0Max_Pool_5/Max:output:0Max_Pool_7/Max:output:0%Pool_Concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Pool_Concatenate/concat
Concatenate_Dropout/IdentityIdentity Pool_Concatenate/concat:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Concatenate_Dropout/Identity¯
 Prediction/MatMul/ReadVariableOpReadVariableOp)prediction_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 Prediction/MatMul/ReadVariableOp³
Prediction/MatMulMatMul%Concatenate_Dropout/Identity:output:0(Prediction/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Prediction/MatMul­
!Prediction/BiasAdd/ReadVariableOpReadVariableOp*prediction_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!Prediction/BiasAdd/ReadVariableOp­
Prediction/BiasAddBiasAddPrediction/MatMul:product:0)Prediction/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Prediction/BiasAdd
Prediction/SigmoidSigmoidPrediction/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Prediction/Sigmoid
IdentityIdentityPrediction/Sigmoid:y:0*^Batch_Norm_3_128/batchnorm/ReadVariableOp,^Batch_Norm_3_128/batchnorm/ReadVariableOp_1,^Batch_Norm_3_128/batchnorm/ReadVariableOp_2.^Batch_Norm_3_128/batchnorm/mul/ReadVariableOp*^Batch_Norm_3_256/batchnorm/ReadVariableOp,^Batch_Norm_3_256/batchnorm/ReadVariableOp_1,^Batch_Norm_3_256/batchnorm/ReadVariableOp_2.^Batch_Norm_3_256/batchnorm/mul/ReadVariableOp*^Batch_Norm_5_128/batchnorm/ReadVariableOp,^Batch_Norm_5_128/batchnorm/ReadVariableOp_1,^Batch_Norm_5_128/batchnorm/ReadVariableOp_2.^Batch_Norm_5_128/batchnorm/mul/ReadVariableOp*^Batch_Norm_5_256/batchnorm/ReadVariableOp,^Batch_Norm_5_256/batchnorm/ReadVariableOp_1,^Batch_Norm_5_256/batchnorm/ReadVariableOp_2.^Batch_Norm_5_256/batchnorm/mul/ReadVariableOp*^Batch_Norm_7_128/batchnorm/ReadVariableOp,^Batch_Norm_7_128/batchnorm/ReadVariableOp_1,^Batch_Norm_7_128/batchnorm/ReadVariableOp_2.^Batch_Norm_7_128/batchnorm/mul/ReadVariableOp*^Batch_Norm_7_256/batchnorm/ReadVariableOp,^Batch_Norm_7_256/batchnorm/ReadVariableOp_1,^Batch_Norm_7_256/batchnorm/ReadVariableOp_2.^Batch_Norm_7_256/batchnorm/mul/ReadVariableOp$^Conv1D_3_128/BiasAdd/ReadVariableOp0^Conv1D_3_128/conv1d/ExpandDims_1/ReadVariableOp$^Conv1D_3_256/BiasAdd/ReadVariableOp0^Conv1D_3_256/conv1d/ExpandDims_1/ReadVariableOp$^Conv1D_5_128/BiasAdd/ReadVariableOp0^Conv1D_5_128/conv1d/ExpandDims_1/ReadVariableOp$^Conv1D_5_256/BiasAdd/ReadVariableOp0^Conv1D_5_256/conv1d/ExpandDims_1/ReadVariableOp$^Conv1D_7_128/BiasAdd/ReadVariableOp0^Conv1D_7_128/conv1d/ExpandDims_1/ReadVariableOp$^Conv1D_7_256/BiasAdd/ReadVariableOp0^Conv1D_7_256/conv1d/ExpandDims_1/ReadVariableOp"^Prediction/BiasAdd/ReadVariableOp!^Prediction/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapeso
m:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2V
)Batch_Norm_3_128/batchnorm/ReadVariableOp)Batch_Norm_3_128/batchnorm/ReadVariableOp2Z
+Batch_Norm_3_128/batchnorm/ReadVariableOp_1+Batch_Norm_3_128/batchnorm/ReadVariableOp_12Z
+Batch_Norm_3_128/batchnorm/ReadVariableOp_2+Batch_Norm_3_128/batchnorm/ReadVariableOp_22^
-Batch_Norm_3_128/batchnorm/mul/ReadVariableOp-Batch_Norm_3_128/batchnorm/mul/ReadVariableOp2V
)Batch_Norm_3_256/batchnorm/ReadVariableOp)Batch_Norm_3_256/batchnorm/ReadVariableOp2Z
+Batch_Norm_3_256/batchnorm/ReadVariableOp_1+Batch_Norm_3_256/batchnorm/ReadVariableOp_12Z
+Batch_Norm_3_256/batchnorm/ReadVariableOp_2+Batch_Norm_3_256/batchnorm/ReadVariableOp_22^
-Batch_Norm_3_256/batchnorm/mul/ReadVariableOp-Batch_Norm_3_256/batchnorm/mul/ReadVariableOp2V
)Batch_Norm_5_128/batchnorm/ReadVariableOp)Batch_Norm_5_128/batchnorm/ReadVariableOp2Z
+Batch_Norm_5_128/batchnorm/ReadVariableOp_1+Batch_Norm_5_128/batchnorm/ReadVariableOp_12Z
+Batch_Norm_5_128/batchnorm/ReadVariableOp_2+Batch_Norm_5_128/batchnorm/ReadVariableOp_22^
-Batch_Norm_5_128/batchnorm/mul/ReadVariableOp-Batch_Norm_5_128/batchnorm/mul/ReadVariableOp2V
)Batch_Norm_5_256/batchnorm/ReadVariableOp)Batch_Norm_5_256/batchnorm/ReadVariableOp2Z
+Batch_Norm_5_256/batchnorm/ReadVariableOp_1+Batch_Norm_5_256/batchnorm/ReadVariableOp_12Z
+Batch_Norm_5_256/batchnorm/ReadVariableOp_2+Batch_Norm_5_256/batchnorm/ReadVariableOp_22^
-Batch_Norm_5_256/batchnorm/mul/ReadVariableOp-Batch_Norm_5_256/batchnorm/mul/ReadVariableOp2V
)Batch_Norm_7_128/batchnorm/ReadVariableOp)Batch_Norm_7_128/batchnorm/ReadVariableOp2Z
+Batch_Norm_7_128/batchnorm/ReadVariableOp_1+Batch_Norm_7_128/batchnorm/ReadVariableOp_12Z
+Batch_Norm_7_128/batchnorm/ReadVariableOp_2+Batch_Norm_7_128/batchnorm/ReadVariableOp_22^
-Batch_Norm_7_128/batchnorm/mul/ReadVariableOp-Batch_Norm_7_128/batchnorm/mul/ReadVariableOp2V
)Batch_Norm_7_256/batchnorm/ReadVariableOp)Batch_Norm_7_256/batchnorm/ReadVariableOp2Z
+Batch_Norm_7_256/batchnorm/ReadVariableOp_1+Batch_Norm_7_256/batchnorm/ReadVariableOp_12Z
+Batch_Norm_7_256/batchnorm/ReadVariableOp_2+Batch_Norm_7_256/batchnorm/ReadVariableOp_22^
-Batch_Norm_7_256/batchnorm/mul/ReadVariableOp-Batch_Norm_7_256/batchnorm/mul/ReadVariableOp2J
#Conv1D_3_128/BiasAdd/ReadVariableOp#Conv1D_3_128/BiasAdd/ReadVariableOp2b
/Conv1D_3_128/conv1d/ExpandDims_1/ReadVariableOp/Conv1D_3_128/conv1d/ExpandDims_1/ReadVariableOp2J
#Conv1D_3_256/BiasAdd/ReadVariableOp#Conv1D_3_256/BiasAdd/ReadVariableOp2b
/Conv1D_3_256/conv1d/ExpandDims_1/ReadVariableOp/Conv1D_3_256/conv1d/ExpandDims_1/ReadVariableOp2J
#Conv1D_5_128/BiasAdd/ReadVariableOp#Conv1D_5_128/BiasAdd/ReadVariableOp2b
/Conv1D_5_128/conv1d/ExpandDims_1/ReadVariableOp/Conv1D_5_128/conv1d/ExpandDims_1/ReadVariableOp2J
#Conv1D_5_256/BiasAdd/ReadVariableOp#Conv1D_5_256/BiasAdd/ReadVariableOp2b
/Conv1D_5_256/conv1d/ExpandDims_1/ReadVariableOp/Conv1D_5_256/conv1d/ExpandDims_1/ReadVariableOp2J
#Conv1D_7_128/BiasAdd/ReadVariableOp#Conv1D_7_128/BiasAdd/ReadVariableOp2b
/Conv1D_7_128/conv1d/ExpandDims_1/ReadVariableOp/Conv1D_7_128/conv1d/ExpandDims_1/ReadVariableOp2J
#Conv1D_7_256/BiasAdd/ReadVariableOp#Conv1D_7_256/BiasAdd/ReadVariableOp2b
/Conv1D_7_256/conv1d/ExpandDims_1/ReadVariableOp/Conv1D_7_256/conv1d/ExpandDims_1/ReadVariableOp2F
!Prediction/BiasAdd/ReadVariableOp!Prediction/BiasAdd/ReadVariableOp2D
 Prediction/MatMul/ReadVariableOp Prediction/MatMul/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü
 
L__inference_Conv1D_5_128_layer_call_and_return_conditional_losses_1916622597

inputsC
+conv1d_expanddims_1_readvariableop_resource:.
biasadd_readvariableop_resource:	

identity_1¢BiasAdd/ReadVariableOp¢"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims/dim 
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
conv1d/ExpandDimsº
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim¹
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv1d/ExpandDims_1À
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAddo
SigmoidSigmoidBiasAdd:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
Sigmoidp
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
muli
IdentityIdentitymul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

IdentityÕ
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1916622590*V
_output_shapesD
B:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	IdentityN¶

Identity_1IdentityIdentityN:output:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í
¥
1__inference_Conv1D_3_256_layer_call_fn_1916624693

inputs
unknown:
	unknown_0:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_Conv1D_3_256_layer_call_and_return_conditional_losses_19166227322
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
k

L__inference_Tree_Encoder_layer_call_and_return_conditional_losses_1916622803

inputs/
conv1d_7_128_1916622571:&
conv1d_7_128_1916622573:	/
conv1d_5_128_1916622598:&
conv1d_5_128_1916622600:	/
conv1d_3_128_1916622625:&
conv1d_3_128_1916622627:	*
batch_norm_7_128_1916622630:	*
batch_norm_7_128_1916622632:	*
batch_norm_7_128_1916622634:	*
batch_norm_7_128_1916622636:	*
batch_norm_5_128_1916622639:	*
batch_norm_5_128_1916622641:	*
batch_norm_5_128_1916622643:	*
batch_norm_5_128_1916622645:	*
batch_norm_3_128_1916622648:	*
batch_norm_3_128_1916622650:	*
batch_norm_3_128_1916622652:	*
batch_norm_3_128_1916622654:	/
conv1d_7_256_1916622679:&
conv1d_7_256_1916622681:	/
conv1d_5_256_1916622706:&
conv1d_5_256_1916622708:	/
conv1d_3_256_1916622733:&
conv1d_3_256_1916622735:	*
batch_norm_7_256_1916622738:	*
batch_norm_7_256_1916622740:	*
batch_norm_7_256_1916622742:	*
batch_norm_7_256_1916622744:	*
batch_norm_5_256_1916622747:	*
batch_norm_5_256_1916622749:	*
batch_norm_5_256_1916622751:	*
batch_norm_5_256_1916622753:	*
batch_norm_3_256_1916622756:	*
batch_norm_3_256_1916622758:	*
batch_norm_3_256_1916622760:	*
batch_norm_3_256_1916622762:	(
prediction_1916622797:	#
prediction_1916622799:
identity¢(Batch_Norm_3_128/StatefulPartitionedCall¢(Batch_Norm_3_256/StatefulPartitionedCall¢(Batch_Norm_5_128/StatefulPartitionedCall¢(Batch_Norm_5_256/StatefulPartitionedCall¢(Batch_Norm_7_128/StatefulPartitionedCall¢(Batch_Norm_7_256/StatefulPartitionedCall¢$Conv1D_3_128/StatefulPartitionedCall¢$Conv1D_3_256/StatefulPartitionedCall¢$Conv1D_5_128/StatefulPartitionedCall¢$Conv1D_5_256/StatefulPartitionedCall¢$Conv1D_7_128/StatefulPartitionedCall¢$Conv1D_7_256/StatefulPartitionedCall¢"Prediction/StatefulPartitionedCall
!Embedding_Dropout/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_Embedding_Dropout_layer_call_and_return_conditional_losses_19166225472#
!Embedding_Dropout/PartitionedCallé
$Conv1D_7_128/StatefulPartitionedCallStatefulPartitionedCall*Embedding_Dropout/PartitionedCall:output:0conv1d_7_128_1916622571conv1d_7_128_1916622573*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_Conv1D_7_128_layer_call_and_return_conditional_losses_19166225702&
$Conv1D_7_128/StatefulPartitionedCallé
$Conv1D_5_128/StatefulPartitionedCallStatefulPartitionedCall*Embedding_Dropout/PartitionedCall:output:0conv1d_5_128_1916622598conv1d_5_128_1916622600*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_Conv1D_5_128_layer_call_and_return_conditional_losses_19166225972&
$Conv1D_5_128/StatefulPartitionedCallé
$Conv1D_3_128/StatefulPartitionedCallStatefulPartitionedCall*Embedding_Dropout/PartitionedCall:output:0conv1d_3_128_1916622625conv1d_3_128_1916622627*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_Conv1D_3_128_layer_call_and_return_conditional_losses_19166226242&
$Conv1D_3_128/StatefulPartitionedCall¾
(Batch_Norm_7_128/StatefulPartitionedCallStatefulPartitionedCall-Conv1D_7_128/StatefulPartitionedCall:output:0batch_norm_7_128_1916622630batch_norm_7_128_1916622632batch_norm_7_128_1916622634batch_norm_7_128_1916622636*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_Batch_Norm_7_128_layer_call_and_return_conditional_losses_19166218722*
(Batch_Norm_7_128/StatefulPartitionedCall¾
(Batch_Norm_5_128/StatefulPartitionedCallStatefulPartitionedCall-Conv1D_5_128/StatefulPartitionedCall:output:0batch_norm_5_128_1916622639batch_norm_5_128_1916622641batch_norm_5_128_1916622643batch_norm_5_128_1916622645*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_Batch_Norm_5_128_layer_call_and_return_conditional_losses_19166217102*
(Batch_Norm_5_128/StatefulPartitionedCall¾
(Batch_Norm_3_128/StatefulPartitionedCallStatefulPartitionedCall-Conv1D_3_128/StatefulPartitionedCall:output:0batch_norm_3_128_1916622648batch_norm_3_128_1916622650batch_norm_3_128_1916622652batch_norm_3_128_1916622654*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_Batch_Norm_3_128_layer_call_and_return_conditional_losses_19166215482*
(Batch_Norm_3_128/StatefulPartitionedCallð
$Conv1D_7_256/StatefulPartitionedCallStatefulPartitionedCall1Batch_Norm_7_128/StatefulPartitionedCall:output:0conv1d_7_256_1916622679conv1d_7_256_1916622681*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_Conv1D_7_256_layer_call_and_return_conditional_losses_19166226782&
$Conv1D_7_256/StatefulPartitionedCallð
$Conv1D_5_256/StatefulPartitionedCallStatefulPartitionedCall1Batch_Norm_5_128/StatefulPartitionedCall:output:0conv1d_5_256_1916622706conv1d_5_256_1916622708*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_Conv1D_5_256_layer_call_and_return_conditional_losses_19166227052&
$Conv1D_5_256/StatefulPartitionedCallð
$Conv1D_3_256/StatefulPartitionedCallStatefulPartitionedCall1Batch_Norm_3_128/StatefulPartitionedCall:output:0conv1d_3_256_1916622733conv1d_3_256_1916622735*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_Conv1D_3_256_layer_call_and_return_conditional_losses_19166227322&
$Conv1D_3_256/StatefulPartitionedCall¾
(Batch_Norm_7_256/StatefulPartitionedCallStatefulPartitionedCall-Conv1D_7_256/StatefulPartitionedCall:output:0batch_norm_7_256_1916622738batch_norm_7_256_1916622740batch_norm_7_256_1916622742batch_norm_7_256_1916622744*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_Batch_Norm_7_256_layer_call_and_return_conditional_losses_19166223582*
(Batch_Norm_7_256/StatefulPartitionedCall¾
(Batch_Norm_5_256/StatefulPartitionedCallStatefulPartitionedCall-Conv1D_5_256/StatefulPartitionedCall:output:0batch_norm_5_256_1916622747batch_norm_5_256_1916622749batch_norm_5_256_1916622751batch_norm_5_256_1916622753*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_Batch_Norm_5_256_layer_call_and_return_conditional_losses_19166221962*
(Batch_Norm_5_256/StatefulPartitionedCall¾
(Batch_Norm_3_256/StatefulPartitionedCallStatefulPartitionedCall-Conv1D_3_256/StatefulPartitionedCall:output:0batch_norm_3_256_1916622756batch_norm_3_256_1916622758batch_norm_3_256_1916622760batch_norm_3_256_1916622762*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_Batch_Norm_3_256_layer_call_and_return_conditional_losses_19166220342*
(Batch_Norm_3_256/StatefulPartitionedCall
Max_Pool_3/PartitionedCallPartitionedCall1Batch_Norm_3_256/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Max_Pool_3_layer_call_and_return_conditional_losses_19166225032
Max_Pool_3/PartitionedCall
Max_Pool_5/PartitionedCallPartitionedCall1Batch_Norm_5_256/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Max_Pool_5_layer_call_and_return_conditional_losses_19166225162
Max_Pool_5/PartitionedCall
Max_Pool_7/PartitionedCallPartitionedCall1Batch_Norm_7_256/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Max_Pool_7_layer_call_and_return_conditional_losses_19166225292
Max_Pool_7/PartitionedCallÝ
 Pool_Concatenate/PartitionedCallPartitionedCall#Max_Pool_3/PartitionedCall:output:0#Max_Pool_5/PartitionedCall:output:0#Max_Pool_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_Pool_Concatenate_layer_call_and_return_conditional_losses_19166227762"
 Pool_Concatenate/PartitionedCall 
#Concatenate_Dropout/PartitionedCallPartitionedCall)Pool_Concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_Concatenate_Dropout_layer_call_and_return_conditional_losses_19166227832%
#Concatenate_Dropout/PartitionedCallÓ
"Prediction/StatefulPartitionedCallStatefulPartitionedCall,Concatenate_Dropout/PartitionedCall:output:0prediction_1916622797prediction_1916622799*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Prediction_layer_call_and_return_conditional_losses_19166227962$
"Prediction/StatefulPartitionedCall
IdentityIdentity+Prediction/StatefulPartitionedCall:output:0)^Batch_Norm_3_128/StatefulPartitionedCall)^Batch_Norm_3_256/StatefulPartitionedCall)^Batch_Norm_5_128/StatefulPartitionedCall)^Batch_Norm_5_256/StatefulPartitionedCall)^Batch_Norm_7_128/StatefulPartitionedCall)^Batch_Norm_7_256/StatefulPartitionedCall%^Conv1D_3_128/StatefulPartitionedCall%^Conv1D_3_256/StatefulPartitionedCall%^Conv1D_5_128/StatefulPartitionedCall%^Conv1D_5_256/StatefulPartitionedCall%^Conv1D_7_128/StatefulPartitionedCall%^Conv1D_7_256/StatefulPartitionedCall#^Prediction/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapeso
m:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2T
(Batch_Norm_3_128/StatefulPartitionedCall(Batch_Norm_3_128/StatefulPartitionedCall2T
(Batch_Norm_3_256/StatefulPartitionedCall(Batch_Norm_3_256/StatefulPartitionedCall2T
(Batch_Norm_5_128/StatefulPartitionedCall(Batch_Norm_5_128/StatefulPartitionedCall2T
(Batch_Norm_5_256/StatefulPartitionedCall(Batch_Norm_5_256/StatefulPartitionedCall2T
(Batch_Norm_7_128/StatefulPartitionedCall(Batch_Norm_7_128/StatefulPartitionedCall2T
(Batch_Norm_7_256/StatefulPartitionedCall(Batch_Norm_7_256/StatefulPartitionedCall2L
$Conv1D_3_128/StatefulPartitionedCall$Conv1D_3_128/StatefulPartitionedCall2L
$Conv1D_3_256/StatefulPartitionedCall$Conv1D_3_256/StatefulPartitionedCall2L
$Conv1D_5_128/StatefulPartitionedCall$Conv1D_5_128/StatefulPartitionedCall2L
$Conv1D_5_256/StatefulPartitionedCall$Conv1D_5_256/StatefulPartitionedCall2L
$Conv1D_7_128/StatefulPartitionedCall$Conv1D_7_128/StatefulPartitionedCall2L
$Conv1D_7_256/StatefulPartitionedCall$Conv1D_7_256/StatefulPartitionedCall2H
"Prediction/StatefulPartitionedCall"Prediction/StatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ
³
P__inference_Batch_Norm_5_256_layer_call_and_return_conditional_losses_1916622196

inputs0
!batchnorm_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	2
#batchnorm_readvariableop_1_resource:	2
#batchnorm_readvariableop_2_resource:	
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1é
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í
¥
1__inference_Conv1D_5_128_layer_call_fn_1916624393

inputs
unknown:
	unknown_0:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_Conv1D_5_128_layer_call_and_return_conditional_losses_19166225972
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾
Ô
5__inference_Batch_Norm_3_256_layer_call_fn_1916624787

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_Batch_Norm_3_256_layer_call_and_return_conditional_losses_19166220342
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¸
*
%__inference__wrapped_model_1916621524

inputs]
Etree_encoder_conv1d_7_128_conv1d_expanddims_1_readvariableop_resource:H
9tree_encoder_conv1d_7_128_biasadd_readvariableop_resource:	]
Etree_encoder_conv1d_5_128_conv1d_expanddims_1_readvariableop_resource:H
9tree_encoder_conv1d_5_128_biasadd_readvariableop_resource:	]
Etree_encoder_conv1d_3_128_conv1d_expanddims_1_readvariableop_resource:H
9tree_encoder_conv1d_3_128_biasadd_readvariableop_resource:	N
?tree_encoder_batch_norm_7_128_batchnorm_readvariableop_resource:	R
Ctree_encoder_batch_norm_7_128_batchnorm_mul_readvariableop_resource:	P
Atree_encoder_batch_norm_7_128_batchnorm_readvariableop_1_resource:	P
Atree_encoder_batch_norm_7_128_batchnorm_readvariableop_2_resource:	N
?tree_encoder_batch_norm_5_128_batchnorm_readvariableop_resource:	R
Ctree_encoder_batch_norm_5_128_batchnorm_mul_readvariableop_resource:	P
Atree_encoder_batch_norm_5_128_batchnorm_readvariableop_1_resource:	P
Atree_encoder_batch_norm_5_128_batchnorm_readvariableop_2_resource:	N
?tree_encoder_batch_norm_3_128_batchnorm_readvariableop_resource:	R
Ctree_encoder_batch_norm_3_128_batchnorm_mul_readvariableop_resource:	P
Atree_encoder_batch_norm_3_128_batchnorm_readvariableop_1_resource:	P
Atree_encoder_batch_norm_3_128_batchnorm_readvariableop_2_resource:	]
Etree_encoder_conv1d_7_256_conv1d_expanddims_1_readvariableop_resource:H
9tree_encoder_conv1d_7_256_biasadd_readvariableop_resource:	]
Etree_encoder_conv1d_5_256_conv1d_expanddims_1_readvariableop_resource:H
9tree_encoder_conv1d_5_256_biasadd_readvariableop_resource:	]
Etree_encoder_conv1d_3_256_conv1d_expanddims_1_readvariableop_resource:H
9tree_encoder_conv1d_3_256_biasadd_readvariableop_resource:	N
?tree_encoder_batch_norm_7_256_batchnorm_readvariableop_resource:	R
Ctree_encoder_batch_norm_7_256_batchnorm_mul_readvariableop_resource:	P
Atree_encoder_batch_norm_7_256_batchnorm_readvariableop_1_resource:	P
Atree_encoder_batch_norm_7_256_batchnorm_readvariableop_2_resource:	N
?tree_encoder_batch_norm_5_256_batchnorm_readvariableop_resource:	R
Ctree_encoder_batch_norm_5_256_batchnorm_mul_readvariableop_resource:	P
Atree_encoder_batch_norm_5_256_batchnorm_readvariableop_1_resource:	P
Atree_encoder_batch_norm_5_256_batchnorm_readvariableop_2_resource:	N
?tree_encoder_batch_norm_3_256_batchnorm_readvariableop_resource:	R
Ctree_encoder_batch_norm_3_256_batchnorm_mul_readvariableop_resource:	P
Atree_encoder_batch_norm_3_256_batchnorm_readvariableop_1_resource:	P
Atree_encoder_batch_norm_3_256_batchnorm_readvariableop_2_resource:	I
6tree_encoder_prediction_matmul_readvariableop_resource:	E
7tree_encoder_prediction_biasadd_readvariableop_resource:
identity¢6Tree_Encoder/Batch_Norm_3_128/batchnorm/ReadVariableOp¢8Tree_Encoder/Batch_Norm_3_128/batchnorm/ReadVariableOp_1¢8Tree_Encoder/Batch_Norm_3_128/batchnorm/ReadVariableOp_2¢:Tree_Encoder/Batch_Norm_3_128/batchnorm/mul/ReadVariableOp¢6Tree_Encoder/Batch_Norm_3_256/batchnorm/ReadVariableOp¢8Tree_Encoder/Batch_Norm_3_256/batchnorm/ReadVariableOp_1¢8Tree_Encoder/Batch_Norm_3_256/batchnorm/ReadVariableOp_2¢:Tree_Encoder/Batch_Norm_3_256/batchnorm/mul/ReadVariableOp¢6Tree_Encoder/Batch_Norm_5_128/batchnorm/ReadVariableOp¢8Tree_Encoder/Batch_Norm_5_128/batchnorm/ReadVariableOp_1¢8Tree_Encoder/Batch_Norm_5_128/batchnorm/ReadVariableOp_2¢:Tree_Encoder/Batch_Norm_5_128/batchnorm/mul/ReadVariableOp¢6Tree_Encoder/Batch_Norm_5_256/batchnorm/ReadVariableOp¢8Tree_Encoder/Batch_Norm_5_256/batchnorm/ReadVariableOp_1¢8Tree_Encoder/Batch_Norm_5_256/batchnorm/ReadVariableOp_2¢:Tree_Encoder/Batch_Norm_5_256/batchnorm/mul/ReadVariableOp¢6Tree_Encoder/Batch_Norm_7_128/batchnorm/ReadVariableOp¢8Tree_Encoder/Batch_Norm_7_128/batchnorm/ReadVariableOp_1¢8Tree_Encoder/Batch_Norm_7_128/batchnorm/ReadVariableOp_2¢:Tree_Encoder/Batch_Norm_7_128/batchnorm/mul/ReadVariableOp¢6Tree_Encoder/Batch_Norm_7_256/batchnorm/ReadVariableOp¢8Tree_Encoder/Batch_Norm_7_256/batchnorm/ReadVariableOp_1¢8Tree_Encoder/Batch_Norm_7_256/batchnorm/ReadVariableOp_2¢:Tree_Encoder/Batch_Norm_7_256/batchnorm/mul/ReadVariableOp¢0Tree_Encoder/Conv1D_3_128/BiasAdd/ReadVariableOp¢<Tree_Encoder/Conv1D_3_128/conv1d/ExpandDims_1/ReadVariableOp¢0Tree_Encoder/Conv1D_3_256/BiasAdd/ReadVariableOp¢<Tree_Encoder/Conv1D_3_256/conv1d/ExpandDims_1/ReadVariableOp¢0Tree_Encoder/Conv1D_5_128/BiasAdd/ReadVariableOp¢<Tree_Encoder/Conv1D_5_128/conv1d/ExpandDims_1/ReadVariableOp¢0Tree_Encoder/Conv1D_5_256/BiasAdd/ReadVariableOp¢<Tree_Encoder/Conv1D_5_256/conv1d/ExpandDims_1/ReadVariableOp¢0Tree_Encoder/Conv1D_7_128/BiasAdd/ReadVariableOp¢<Tree_Encoder/Conv1D_7_128/conv1d/ExpandDims_1/ReadVariableOp¢0Tree_Encoder/Conv1D_7_256/BiasAdd/ReadVariableOp¢<Tree_Encoder/Conv1D_7_256/conv1d/ExpandDims_1/ReadVariableOp¢.Tree_Encoder/Prediction/BiasAdd/ReadVariableOp¢-Tree_Encoder/Prediction/MatMul/ReadVariableOp¦
'Tree_Encoder/Embedding_Dropout/IdentityIdentityinputs*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2)
'Tree_Encoder/Embedding_Dropout/Identity­
/Tree_Encoder/Conv1D_7_128/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ21
/Tree_Encoder/Conv1D_7_128/conv1d/ExpandDims/dim
+Tree_Encoder/Conv1D_7_128/conv1d/ExpandDims
ExpandDims0Tree_Encoder/Embedding_Dropout/Identity:output:08Tree_Encoder/Conv1D_7_128/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2-
+Tree_Encoder/Conv1D_7_128/conv1d/ExpandDims
<Tree_Encoder/Conv1D_7_128/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEtree_encoder_conv1d_7_128_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02>
<Tree_Encoder/Conv1D_7_128/conv1d/ExpandDims_1/ReadVariableOp¨
1Tree_Encoder/Conv1D_7_128/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1Tree_Encoder/Conv1D_7_128/conv1d/ExpandDims_1/dim¡
-Tree_Encoder/Conv1D_7_128/conv1d/ExpandDims_1
ExpandDimsDTree_Encoder/Conv1D_7_128/conv1d/ExpandDims_1/ReadVariableOp:value:0:Tree_Encoder/Conv1D_7_128/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2/
-Tree_Encoder/Conv1D_7_128/conv1d/ExpandDims_1¨
 Tree_Encoder/Conv1D_7_128/conv1dConv2D4Tree_Encoder/Conv1D_7_128/conv1d/ExpandDims:output:06Tree_Encoder/Conv1D_7_128/conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2"
 Tree_Encoder/Conv1D_7_128/conv1dê
(Tree_Encoder/Conv1D_7_128/conv1d/SqueezeSqueeze)Tree_Encoder/Conv1D_7_128/conv1d:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2*
(Tree_Encoder/Conv1D_7_128/conv1d/SqueezeÛ
0Tree_Encoder/Conv1D_7_128/BiasAdd/ReadVariableOpReadVariableOp9tree_encoder_conv1d_7_128_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype022
0Tree_Encoder/Conv1D_7_128/BiasAdd/ReadVariableOpþ
!Tree_Encoder/Conv1D_7_128/BiasAddBiasAdd1Tree_Encoder/Conv1D_7_128/conv1d/Squeeze:output:08Tree_Encoder/Conv1D_7_128/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2#
!Tree_Encoder/Conv1D_7_128/BiasAdd½
!Tree_Encoder/Conv1D_7_128/SigmoidSigmoid*Tree_Encoder/Conv1D_7_128/BiasAdd:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2#
!Tree_Encoder/Conv1D_7_128/SigmoidØ
Tree_Encoder/Conv1D_7_128/mulMul*Tree_Encoder/Conv1D_7_128/BiasAdd:output:0%Tree_Encoder/Conv1D_7_128/Sigmoid:y:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tree_Encoder/Conv1D_7_128/mul·
"Tree_Encoder/Conv1D_7_128/IdentityIdentity!Tree_Encoder/Conv1D_7_128/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2$
"Tree_Encoder/Conv1D_7_128/Identity½
#Tree_Encoder/Conv1D_7_128/IdentityN	IdentityN!Tree_Encoder/Conv1D_7_128/mul:z:0*Tree_Encoder/Conv1D_7_128/BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1916621320*V
_output_shapesD
B:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2%
#Tree_Encoder/Conv1D_7_128/IdentityN­
/Tree_Encoder/Conv1D_5_128/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ21
/Tree_Encoder/Conv1D_5_128/conv1d/ExpandDims/dim
+Tree_Encoder/Conv1D_5_128/conv1d/ExpandDims
ExpandDims0Tree_Encoder/Embedding_Dropout/Identity:output:08Tree_Encoder/Conv1D_5_128/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2-
+Tree_Encoder/Conv1D_5_128/conv1d/ExpandDims
<Tree_Encoder/Conv1D_5_128/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEtree_encoder_conv1d_5_128_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02>
<Tree_Encoder/Conv1D_5_128/conv1d/ExpandDims_1/ReadVariableOp¨
1Tree_Encoder/Conv1D_5_128/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1Tree_Encoder/Conv1D_5_128/conv1d/ExpandDims_1/dim¡
-Tree_Encoder/Conv1D_5_128/conv1d/ExpandDims_1
ExpandDimsDTree_Encoder/Conv1D_5_128/conv1d/ExpandDims_1/ReadVariableOp:value:0:Tree_Encoder/Conv1D_5_128/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2/
-Tree_Encoder/Conv1D_5_128/conv1d/ExpandDims_1¨
 Tree_Encoder/Conv1D_5_128/conv1dConv2D4Tree_Encoder/Conv1D_5_128/conv1d/ExpandDims:output:06Tree_Encoder/Conv1D_5_128/conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2"
 Tree_Encoder/Conv1D_5_128/conv1dê
(Tree_Encoder/Conv1D_5_128/conv1d/SqueezeSqueeze)Tree_Encoder/Conv1D_5_128/conv1d:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2*
(Tree_Encoder/Conv1D_5_128/conv1d/SqueezeÛ
0Tree_Encoder/Conv1D_5_128/BiasAdd/ReadVariableOpReadVariableOp9tree_encoder_conv1d_5_128_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype022
0Tree_Encoder/Conv1D_5_128/BiasAdd/ReadVariableOpþ
!Tree_Encoder/Conv1D_5_128/BiasAddBiasAdd1Tree_Encoder/Conv1D_5_128/conv1d/Squeeze:output:08Tree_Encoder/Conv1D_5_128/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2#
!Tree_Encoder/Conv1D_5_128/BiasAdd½
!Tree_Encoder/Conv1D_5_128/SigmoidSigmoid*Tree_Encoder/Conv1D_5_128/BiasAdd:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2#
!Tree_Encoder/Conv1D_5_128/SigmoidØ
Tree_Encoder/Conv1D_5_128/mulMul*Tree_Encoder/Conv1D_5_128/BiasAdd:output:0%Tree_Encoder/Conv1D_5_128/Sigmoid:y:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tree_Encoder/Conv1D_5_128/mul·
"Tree_Encoder/Conv1D_5_128/IdentityIdentity!Tree_Encoder/Conv1D_5_128/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2$
"Tree_Encoder/Conv1D_5_128/Identity½
#Tree_Encoder/Conv1D_5_128/IdentityN	IdentityN!Tree_Encoder/Conv1D_5_128/mul:z:0*Tree_Encoder/Conv1D_5_128/BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1916621337*V
_output_shapesD
B:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2%
#Tree_Encoder/Conv1D_5_128/IdentityN­
/Tree_Encoder/Conv1D_3_128/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ21
/Tree_Encoder/Conv1D_3_128/conv1d/ExpandDims/dim
+Tree_Encoder/Conv1D_3_128/conv1d/ExpandDims
ExpandDims0Tree_Encoder/Embedding_Dropout/Identity:output:08Tree_Encoder/Conv1D_3_128/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2-
+Tree_Encoder/Conv1D_3_128/conv1d/ExpandDims
<Tree_Encoder/Conv1D_3_128/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEtree_encoder_conv1d_3_128_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02>
<Tree_Encoder/Conv1D_3_128/conv1d/ExpandDims_1/ReadVariableOp¨
1Tree_Encoder/Conv1D_3_128/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1Tree_Encoder/Conv1D_3_128/conv1d/ExpandDims_1/dim¡
-Tree_Encoder/Conv1D_3_128/conv1d/ExpandDims_1
ExpandDimsDTree_Encoder/Conv1D_3_128/conv1d/ExpandDims_1/ReadVariableOp:value:0:Tree_Encoder/Conv1D_3_128/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2/
-Tree_Encoder/Conv1D_3_128/conv1d/ExpandDims_1¨
 Tree_Encoder/Conv1D_3_128/conv1dConv2D4Tree_Encoder/Conv1D_3_128/conv1d/ExpandDims:output:06Tree_Encoder/Conv1D_3_128/conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2"
 Tree_Encoder/Conv1D_3_128/conv1dê
(Tree_Encoder/Conv1D_3_128/conv1d/SqueezeSqueeze)Tree_Encoder/Conv1D_3_128/conv1d:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2*
(Tree_Encoder/Conv1D_3_128/conv1d/SqueezeÛ
0Tree_Encoder/Conv1D_3_128/BiasAdd/ReadVariableOpReadVariableOp9tree_encoder_conv1d_3_128_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype022
0Tree_Encoder/Conv1D_3_128/BiasAdd/ReadVariableOpþ
!Tree_Encoder/Conv1D_3_128/BiasAddBiasAdd1Tree_Encoder/Conv1D_3_128/conv1d/Squeeze:output:08Tree_Encoder/Conv1D_3_128/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2#
!Tree_Encoder/Conv1D_3_128/BiasAdd½
!Tree_Encoder/Conv1D_3_128/SigmoidSigmoid*Tree_Encoder/Conv1D_3_128/BiasAdd:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2#
!Tree_Encoder/Conv1D_3_128/SigmoidØ
Tree_Encoder/Conv1D_3_128/mulMul*Tree_Encoder/Conv1D_3_128/BiasAdd:output:0%Tree_Encoder/Conv1D_3_128/Sigmoid:y:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tree_Encoder/Conv1D_3_128/mul·
"Tree_Encoder/Conv1D_3_128/IdentityIdentity!Tree_Encoder/Conv1D_3_128/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2$
"Tree_Encoder/Conv1D_3_128/Identity½
#Tree_Encoder/Conv1D_3_128/IdentityN	IdentityN!Tree_Encoder/Conv1D_3_128/mul:z:0*Tree_Encoder/Conv1D_3_128/BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1916621354*V
_output_shapesD
B:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2%
#Tree_Encoder/Conv1D_3_128/IdentityNí
6Tree_Encoder/Batch_Norm_7_128/batchnorm/ReadVariableOpReadVariableOp?tree_encoder_batch_norm_7_128_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype028
6Tree_Encoder/Batch_Norm_7_128/batchnorm/ReadVariableOp£
-Tree_Encoder/Batch_Norm_7_128/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2/
-Tree_Encoder/Batch_Norm_7_128/batchnorm/add/y
+Tree_Encoder/Batch_Norm_7_128/batchnorm/addAddV2>Tree_Encoder/Batch_Norm_7_128/batchnorm/ReadVariableOp:value:06Tree_Encoder/Batch_Norm_7_128/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2-
+Tree_Encoder/Batch_Norm_7_128/batchnorm/add¾
-Tree_Encoder/Batch_Norm_7_128/batchnorm/RsqrtRsqrt/Tree_Encoder/Batch_Norm_7_128/batchnorm/add:z:0*
T0*
_output_shapes	
:2/
-Tree_Encoder/Batch_Norm_7_128/batchnorm/Rsqrtù
:Tree_Encoder/Batch_Norm_7_128/batchnorm/mul/ReadVariableOpReadVariableOpCtree_encoder_batch_norm_7_128_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02<
:Tree_Encoder/Batch_Norm_7_128/batchnorm/mul/ReadVariableOpþ
+Tree_Encoder/Batch_Norm_7_128/batchnorm/mulMul1Tree_Encoder/Batch_Norm_7_128/batchnorm/Rsqrt:y:0BTree_Encoder/Batch_Norm_7_128/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2-
+Tree_Encoder/Batch_Norm_7_128/batchnorm/mul
-Tree_Encoder/Batch_Norm_7_128/batchnorm/mul_1Mul,Tree_Encoder/Conv1D_7_128/IdentityN:output:0/Tree_Encoder/Batch_Norm_7_128/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2/
-Tree_Encoder/Batch_Norm_7_128/batchnorm/mul_1ó
8Tree_Encoder/Batch_Norm_7_128/batchnorm/ReadVariableOp_1ReadVariableOpAtree_encoder_batch_norm_7_128_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02:
8Tree_Encoder/Batch_Norm_7_128/batchnorm/ReadVariableOp_1þ
-Tree_Encoder/Batch_Norm_7_128/batchnorm/mul_2Mul@Tree_Encoder/Batch_Norm_7_128/batchnorm/ReadVariableOp_1:value:0/Tree_Encoder/Batch_Norm_7_128/batchnorm/mul:z:0*
T0*
_output_shapes	
:2/
-Tree_Encoder/Batch_Norm_7_128/batchnorm/mul_2ó
8Tree_Encoder/Batch_Norm_7_128/batchnorm/ReadVariableOp_2ReadVariableOpAtree_encoder_batch_norm_7_128_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02:
8Tree_Encoder/Batch_Norm_7_128/batchnorm/ReadVariableOp_2ü
+Tree_Encoder/Batch_Norm_7_128/batchnorm/subSub@Tree_Encoder/Batch_Norm_7_128/batchnorm/ReadVariableOp_2:value:01Tree_Encoder/Batch_Norm_7_128/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2-
+Tree_Encoder/Batch_Norm_7_128/batchnorm/sub
-Tree_Encoder/Batch_Norm_7_128/batchnorm/add_1AddV21Tree_Encoder/Batch_Norm_7_128/batchnorm/mul_1:z:0/Tree_Encoder/Batch_Norm_7_128/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2/
-Tree_Encoder/Batch_Norm_7_128/batchnorm/add_1í
6Tree_Encoder/Batch_Norm_5_128/batchnorm/ReadVariableOpReadVariableOp?tree_encoder_batch_norm_5_128_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype028
6Tree_Encoder/Batch_Norm_5_128/batchnorm/ReadVariableOp£
-Tree_Encoder/Batch_Norm_5_128/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2/
-Tree_Encoder/Batch_Norm_5_128/batchnorm/add/y
+Tree_Encoder/Batch_Norm_5_128/batchnorm/addAddV2>Tree_Encoder/Batch_Norm_5_128/batchnorm/ReadVariableOp:value:06Tree_Encoder/Batch_Norm_5_128/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2-
+Tree_Encoder/Batch_Norm_5_128/batchnorm/add¾
-Tree_Encoder/Batch_Norm_5_128/batchnorm/RsqrtRsqrt/Tree_Encoder/Batch_Norm_5_128/batchnorm/add:z:0*
T0*
_output_shapes	
:2/
-Tree_Encoder/Batch_Norm_5_128/batchnorm/Rsqrtù
:Tree_Encoder/Batch_Norm_5_128/batchnorm/mul/ReadVariableOpReadVariableOpCtree_encoder_batch_norm_5_128_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02<
:Tree_Encoder/Batch_Norm_5_128/batchnorm/mul/ReadVariableOpþ
+Tree_Encoder/Batch_Norm_5_128/batchnorm/mulMul1Tree_Encoder/Batch_Norm_5_128/batchnorm/Rsqrt:y:0BTree_Encoder/Batch_Norm_5_128/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2-
+Tree_Encoder/Batch_Norm_5_128/batchnorm/mul
-Tree_Encoder/Batch_Norm_5_128/batchnorm/mul_1Mul,Tree_Encoder/Conv1D_5_128/IdentityN:output:0/Tree_Encoder/Batch_Norm_5_128/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2/
-Tree_Encoder/Batch_Norm_5_128/batchnorm/mul_1ó
8Tree_Encoder/Batch_Norm_5_128/batchnorm/ReadVariableOp_1ReadVariableOpAtree_encoder_batch_norm_5_128_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02:
8Tree_Encoder/Batch_Norm_5_128/batchnorm/ReadVariableOp_1þ
-Tree_Encoder/Batch_Norm_5_128/batchnorm/mul_2Mul@Tree_Encoder/Batch_Norm_5_128/batchnorm/ReadVariableOp_1:value:0/Tree_Encoder/Batch_Norm_5_128/batchnorm/mul:z:0*
T0*
_output_shapes	
:2/
-Tree_Encoder/Batch_Norm_5_128/batchnorm/mul_2ó
8Tree_Encoder/Batch_Norm_5_128/batchnorm/ReadVariableOp_2ReadVariableOpAtree_encoder_batch_norm_5_128_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02:
8Tree_Encoder/Batch_Norm_5_128/batchnorm/ReadVariableOp_2ü
+Tree_Encoder/Batch_Norm_5_128/batchnorm/subSub@Tree_Encoder/Batch_Norm_5_128/batchnorm/ReadVariableOp_2:value:01Tree_Encoder/Batch_Norm_5_128/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2-
+Tree_Encoder/Batch_Norm_5_128/batchnorm/sub
-Tree_Encoder/Batch_Norm_5_128/batchnorm/add_1AddV21Tree_Encoder/Batch_Norm_5_128/batchnorm/mul_1:z:0/Tree_Encoder/Batch_Norm_5_128/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2/
-Tree_Encoder/Batch_Norm_5_128/batchnorm/add_1í
6Tree_Encoder/Batch_Norm_3_128/batchnorm/ReadVariableOpReadVariableOp?tree_encoder_batch_norm_3_128_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype028
6Tree_Encoder/Batch_Norm_3_128/batchnorm/ReadVariableOp£
-Tree_Encoder/Batch_Norm_3_128/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2/
-Tree_Encoder/Batch_Norm_3_128/batchnorm/add/y
+Tree_Encoder/Batch_Norm_3_128/batchnorm/addAddV2>Tree_Encoder/Batch_Norm_3_128/batchnorm/ReadVariableOp:value:06Tree_Encoder/Batch_Norm_3_128/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2-
+Tree_Encoder/Batch_Norm_3_128/batchnorm/add¾
-Tree_Encoder/Batch_Norm_3_128/batchnorm/RsqrtRsqrt/Tree_Encoder/Batch_Norm_3_128/batchnorm/add:z:0*
T0*
_output_shapes	
:2/
-Tree_Encoder/Batch_Norm_3_128/batchnorm/Rsqrtù
:Tree_Encoder/Batch_Norm_3_128/batchnorm/mul/ReadVariableOpReadVariableOpCtree_encoder_batch_norm_3_128_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02<
:Tree_Encoder/Batch_Norm_3_128/batchnorm/mul/ReadVariableOpþ
+Tree_Encoder/Batch_Norm_3_128/batchnorm/mulMul1Tree_Encoder/Batch_Norm_3_128/batchnorm/Rsqrt:y:0BTree_Encoder/Batch_Norm_3_128/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2-
+Tree_Encoder/Batch_Norm_3_128/batchnorm/mul
-Tree_Encoder/Batch_Norm_3_128/batchnorm/mul_1Mul,Tree_Encoder/Conv1D_3_128/IdentityN:output:0/Tree_Encoder/Batch_Norm_3_128/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2/
-Tree_Encoder/Batch_Norm_3_128/batchnorm/mul_1ó
8Tree_Encoder/Batch_Norm_3_128/batchnorm/ReadVariableOp_1ReadVariableOpAtree_encoder_batch_norm_3_128_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02:
8Tree_Encoder/Batch_Norm_3_128/batchnorm/ReadVariableOp_1þ
-Tree_Encoder/Batch_Norm_3_128/batchnorm/mul_2Mul@Tree_Encoder/Batch_Norm_3_128/batchnorm/ReadVariableOp_1:value:0/Tree_Encoder/Batch_Norm_3_128/batchnorm/mul:z:0*
T0*
_output_shapes	
:2/
-Tree_Encoder/Batch_Norm_3_128/batchnorm/mul_2ó
8Tree_Encoder/Batch_Norm_3_128/batchnorm/ReadVariableOp_2ReadVariableOpAtree_encoder_batch_norm_3_128_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02:
8Tree_Encoder/Batch_Norm_3_128/batchnorm/ReadVariableOp_2ü
+Tree_Encoder/Batch_Norm_3_128/batchnorm/subSub@Tree_Encoder/Batch_Norm_3_128/batchnorm/ReadVariableOp_2:value:01Tree_Encoder/Batch_Norm_3_128/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2-
+Tree_Encoder/Batch_Norm_3_128/batchnorm/sub
-Tree_Encoder/Batch_Norm_3_128/batchnorm/add_1AddV21Tree_Encoder/Batch_Norm_3_128/batchnorm/mul_1:z:0/Tree_Encoder/Batch_Norm_3_128/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2/
-Tree_Encoder/Batch_Norm_3_128/batchnorm/add_1­
/Tree_Encoder/Conv1D_7_256/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ21
/Tree_Encoder/Conv1D_7_256/conv1d/ExpandDims/dim
+Tree_Encoder/Conv1D_7_256/conv1d/ExpandDims
ExpandDims1Tree_Encoder/Batch_Norm_7_128/batchnorm/add_1:z:08Tree_Encoder/Conv1D_7_256/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2-
+Tree_Encoder/Conv1D_7_256/conv1d/ExpandDims
<Tree_Encoder/Conv1D_7_256/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEtree_encoder_conv1d_7_256_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02>
<Tree_Encoder/Conv1D_7_256/conv1d/ExpandDims_1/ReadVariableOp¨
1Tree_Encoder/Conv1D_7_256/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1Tree_Encoder/Conv1D_7_256/conv1d/ExpandDims_1/dim¡
-Tree_Encoder/Conv1D_7_256/conv1d/ExpandDims_1
ExpandDimsDTree_Encoder/Conv1D_7_256/conv1d/ExpandDims_1/ReadVariableOp:value:0:Tree_Encoder/Conv1D_7_256/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2/
-Tree_Encoder/Conv1D_7_256/conv1d/ExpandDims_1¨
 Tree_Encoder/Conv1D_7_256/conv1dConv2D4Tree_Encoder/Conv1D_7_256/conv1d/ExpandDims:output:06Tree_Encoder/Conv1D_7_256/conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2"
 Tree_Encoder/Conv1D_7_256/conv1dê
(Tree_Encoder/Conv1D_7_256/conv1d/SqueezeSqueeze)Tree_Encoder/Conv1D_7_256/conv1d:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2*
(Tree_Encoder/Conv1D_7_256/conv1d/SqueezeÛ
0Tree_Encoder/Conv1D_7_256/BiasAdd/ReadVariableOpReadVariableOp9tree_encoder_conv1d_7_256_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype022
0Tree_Encoder/Conv1D_7_256/BiasAdd/ReadVariableOpþ
!Tree_Encoder/Conv1D_7_256/BiasAddBiasAdd1Tree_Encoder/Conv1D_7_256/conv1d/Squeeze:output:08Tree_Encoder/Conv1D_7_256/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2#
!Tree_Encoder/Conv1D_7_256/BiasAdd½
!Tree_Encoder/Conv1D_7_256/SigmoidSigmoid*Tree_Encoder/Conv1D_7_256/BiasAdd:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2#
!Tree_Encoder/Conv1D_7_256/SigmoidØ
Tree_Encoder/Conv1D_7_256/mulMul*Tree_Encoder/Conv1D_7_256/BiasAdd:output:0%Tree_Encoder/Conv1D_7_256/Sigmoid:y:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tree_Encoder/Conv1D_7_256/mul·
"Tree_Encoder/Conv1D_7_256/IdentityIdentity!Tree_Encoder/Conv1D_7_256/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2$
"Tree_Encoder/Conv1D_7_256/Identity½
#Tree_Encoder/Conv1D_7_256/IdentityN	IdentityN!Tree_Encoder/Conv1D_7_256/mul:z:0*Tree_Encoder/Conv1D_7_256/BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1916621419*V
_output_shapesD
B:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2%
#Tree_Encoder/Conv1D_7_256/IdentityN­
/Tree_Encoder/Conv1D_5_256/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ21
/Tree_Encoder/Conv1D_5_256/conv1d/ExpandDims/dim
+Tree_Encoder/Conv1D_5_256/conv1d/ExpandDims
ExpandDims1Tree_Encoder/Batch_Norm_5_128/batchnorm/add_1:z:08Tree_Encoder/Conv1D_5_256/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2-
+Tree_Encoder/Conv1D_5_256/conv1d/ExpandDims
<Tree_Encoder/Conv1D_5_256/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEtree_encoder_conv1d_5_256_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02>
<Tree_Encoder/Conv1D_5_256/conv1d/ExpandDims_1/ReadVariableOp¨
1Tree_Encoder/Conv1D_5_256/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1Tree_Encoder/Conv1D_5_256/conv1d/ExpandDims_1/dim¡
-Tree_Encoder/Conv1D_5_256/conv1d/ExpandDims_1
ExpandDimsDTree_Encoder/Conv1D_5_256/conv1d/ExpandDims_1/ReadVariableOp:value:0:Tree_Encoder/Conv1D_5_256/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2/
-Tree_Encoder/Conv1D_5_256/conv1d/ExpandDims_1¨
 Tree_Encoder/Conv1D_5_256/conv1dConv2D4Tree_Encoder/Conv1D_5_256/conv1d/ExpandDims:output:06Tree_Encoder/Conv1D_5_256/conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2"
 Tree_Encoder/Conv1D_5_256/conv1dê
(Tree_Encoder/Conv1D_5_256/conv1d/SqueezeSqueeze)Tree_Encoder/Conv1D_5_256/conv1d:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2*
(Tree_Encoder/Conv1D_5_256/conv1d/SqueezeÛ
0Tree_Encoder/Conv1D_5_256/BiasAdd/ReadVariableOpReadVariableOp9tree_encoder_conv1d_5_256_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype022
0Tree_Encoder/Conv1D_5_256/BiasAdd/ReadVariableOpþ
!Tree_Encoder/Conv1D_5_256/BiasAddBiasAdd1Tree_Encoder/Conv1D_5_256/conv1d/Squeeze:output:08Tree_Encoder/Conv1D_5_256/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2#
!Tree_Encoder/Conv1D_5_256/BiasAdd½
!Tree_Encoder/Conv1D_5_256/SigmoidSigmoid*Tree_Encoder/Conv1D_5_256/BiasAdd:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2#
!Tree_Encoder/Conv1D_5_256/SigmoidØ
Tree_Encoder/Conv1D_5_256/mulMul*Tree_Encoder/Conv1D_5_256/BiasAdd:output:0%Tree_Encoder/Conv1D_5_256/Sigmoid:y:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tree_Encoder/Conv1D_5_256/mul·
"Tree_Encoder/Conv1D_5_256/IdentityIdentity!Tree_Encoder/Conv1D_5_256/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2$
"Tree_Encoder/Conv1D_5_256/Identity½
#Tree_Encoder/Conv1D_5_256/IdentityN	IdentityN!Tree_Encoder/Conv1D_5_256/mul:z:0*Tree_Encoder/Conv1D_5_256/BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1916621436*V
_output_shapesD
B:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2%
#Tree_Encoder/Conv1D_5_256/IdentityN­
/Tree_Encoder/Conv1D_3_256/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ21
/Tree_Encoder/Conv1D_3_256/conv1d/ExpandDims/dim
+Tree_Encoder/Conv1D_3_256/conv1d/ExpandDims
ExpandDims1Tree_Encoder/Batch_Norm_3_128/batchnorm/add_1:z:08Tree_Encoder/Conv1D_3_256/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2-
+Tree_Encoder/Conv1D_3_256/conv1d/ExpandDims
<Tree_Encoder/Conv1D_3_256/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEtree_encoder_conv1d_3_256_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02>
<Tree_Encoder/Conv1D_3_256/conv1d/ExpandDims_1/ReadVariableOp¨
1Tree_Encoder/Conv1D_3_256/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1Tree_Encoder/Conv1D_3_256/conv1d/ExpandDims_1/dim¡
-Tree_Encoder/Conv1D_3_256/conv1d/ExpandDims_1
ExpandDimsDTree_Encoder/Conv1D_3_256/conv1d/ExpandDims_1/ReadVariableOp:value:0:Tree_Encoder/Conv1D_3_256/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2/
-Tree_Encoder/Conv1D_3_256/conv1d/ExpandDims_1¨
 Tree_Encoder/Conv1D_3_256/conv1dConv2D4Tree_Encoder/Conv1D_3_256/conv1d/ExpandDims:output:06Tree_Encoder/Conv1D_3_256/conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2"
 Tree_Encoder/Conv1D_3_256/conv1dê
(Tree_Encoder/Conv1D_3_256/conv1d/SqueezeSqueeze)Tree_Encoder/Conv1D_3_256/conv1d:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2*
(Tree_Encoder/Conv1D_3_256/conv1d/SqueezeÛ
0Tree_Encoder/Conv1D_3_256/BiasAdd/ReadVariableOpReadVariableOp9tree_encoder_conv1d_3_256_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype022
0Tree_Encoder/Conv1D_3_256/BiasAdd/ReadVariableOpþ
!Tree_Encoder/Conv1D_3_256/BiasAddBiasAdd1Tree_Encoder/Conv1D_3_256/conv1d/Squeeze:output:08Tree_Encoder/Conv1D_3_256/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2#
!Tree_Encoder/Conv1D_3_256/BiasAdd½
!Tree_Encoder/Conv1D_3_256/SigmoidSigmoid*Tree_Encoder/Conv1D_3_256/BiasAdd:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2#
!Tree_Encoder/Conv1D_3_256/SigmoidØ
Tree_Encoder/Conv1D_3_256/mulMul*Tree_Encoder/Conv1D_3_256/BiasAdd:output:0%Tree_Encoder/Conv1D_3_256/Sigmoid:y:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tree_Encoder/Conv1D_3_256/mul·
"Tree_Encoder/Conv1D_3_256/IdentityIdentity!Tree_Encoder/Conv1D_3_256/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2$
"Tree_Encoder/Conv1D_3_256/Identity½
#Tree_Encoder/Conv1D_3_256/IdentityN	IdentityN!Tree_Encoder/Conv1D_3_256/mul:z:0*Tree_Encoder/Conv1D_3_256/BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1916621453*V
_output_shapesD
B:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2%
#Tree_Encoder/Conv1D_3_256/IdentityNí
6Tree_Encoder/Batch_Norm_7_256/batchnorm/ReadVariableOpReadVariableOp?tree_encoder_batch_norm_7_256_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype028
6Tree_Encoder/Batch_Norm_7_256/batchnorm/ReadVariableOp£
-Tree_Encoder/Batch_Norm_7_256/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2/
-Tree_Encoder/Batch_Norm_7_256/batchnorm/add/y
+Tree_Encoder/Batch_Norm_7_256/batchnorm/addAddV2>Tree_Encoder/Batch_Norm_7_256/batchnorm/ReadVariableOp:value:06Tree_Encoder/Batch_Norm_7_256/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2-
+Tree_Encoder/Batch_Norm_7_256/batchnorm/add¾
-Tree_Encoder/Batch_Norm_7_256/batchnorm/RsqrtRsqrt/Tree_Encoder/Batch_Norm_7_256/batchnorm/add:z:0*
T0*
_output_shapes	
:2/
-Tree_Encoder/Batch_Norm_7_256/batchnorm/Rsqrtù
:Tree_Encoder/Batch_Norm_7_256/batchnorm/mul/ReadVariableOpReadVariableOpCtree_encoder_batch_norm_7_256_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02<
:Tree_Encoder/Batch_Norm_7_256/batchnorm/mul/ReadVariableOpþ
+Tree_Encoder/Batch_Norm_7_256/batchnorm/mulMul1Tree_Encoder/Batch_Norm_7_256/batchnorm/Rsqrt:y:0BTree_Encoder/Batch_Norm_7_256/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2-
+Tree_Encoder/Batch_Norm_7_256/batchnorm/mul
-Tree_Encoder/Batch_Norm_7_256/batchnorm/mul_1Mul,Tree_Encoder/Conv1D_7_256/IdentityN:output:0/Tree_Encoder/Batch_Norm_7_256/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2/
-Tree_Encoder/Batch_Norm_7_256/batchnorm/mul_1ó
8Tree_Encoder/Batch_Norm_7_256/batchnorm/ReadVariableOp_1ReadVariableOpAtree_encoder_batch_norm_7_256_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02:
8Tree_Encoder/Batch_Norm_7_256/batchnorm/ReadVariableOp_1þ
-Tree_Encoder/Batch_Norm_7_256/batchnorm/mul_2Mul@Tree_Encoder/Batch_Norm_7_256/batchnorm/ReadVariableOp_1:value:0/Tree_Encoder/Batch_Norm_7_256/batchnorm/mul:z:0*
T0*
_output_shapes	
:2/
-Tree_Encoder/Batch_Norm_7_256/batchnorm/mul_2ó
8Tree_Encoder/Batch_Norm_7_256/batchnorm/ReadVariableOp_2ReadVariableOpAtree_encoder_batch_norm_7_256_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02:
8Tree_Encoder/Batch_Norm_7_256/batchnorm/ReadVariableOp_2ü
+Tree_Encoder/Batch_Norm_7_256/batchnorm/subSub@Tree_Encoder/Batch_Norm_7_256/batchnorm/ReadVariableOp_2:value:01Tree_Encoder/Batch_Norm_7_256/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2-
+Tree_Encoder/Batch_Norm_7_256/batchnorm/sub
-Tree_Encoder/Batch_Norm_7_256/batchnorm/add_1AddV21Tree_Encoder/Batch_Norm_7_256/batchnorm/mul_1:z:0/Tree_Encoder/Batch_Norm_7_256/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2/
-Tree_Encoder/Batch_Norm_7_256/batchnorm/add_1í
6Tree_Encoder/Batch_Norm_5_256/batchnorm/ReadVariableOpReadVariableOp?tree_encoder_batch_norm_5_256_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype028
6Tree_Encoder/Batch_Norm_5_256/batchnorm/ReadVariableOp£
-Tree_Encoder/Batch_Norm_5_256/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2/
-Tree_Encoder/Batch_Norm_5_256/batchnorm/add/y
+Tree_Encoder/Batch_Norm_5_256/batchnorm/addAddV2>Tree_Encoder/Batch_Norm_5_256/batchnorm/ReadVariableOp:value:06Tree_Encoder/Batch_Norm_5_256/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2-
+Tree_Encoder/Batch_Norm_5_256/batchnorm/add¾
-Tree_Encoder/Batch_Norm_5_256/batchnorm/RsqrtRsqrt/Tree_Encoder/Batch_Norm_5_256/batchnorm/add:z:0*
T0*
_output_shapes	
:2/
-Tree_Encoder/Batch_Norm_5_256/batchnorm/Rsqrtù
:Tree_Encoder/Batch_Norm_5_256/batchnorm/mul/ReadVariableOpReadVariableOpCtree_encoder_batch_norm_5_256_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02<
:Tree_Encoder/Batch_Norm_5_256/batchnorm/mul/ReadVariableOpþ
+Tree_Encoder/Batch_Norm_5_256/batchnorm/mulMul1Tree_Encoder/Batch_Norm_5_256/batchnorm/Rsqrt:y:0BTree_Encoder/Batch_Norm_5_256/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2-
+Tree_Encoder/Batch_Norm_5_256/batchnorm/mul
-Tree_Encoder/Batch_Norm_5_256/batchnorm/mul_1Mul,Tree_Encoder/Conv1D_5_256/IdentityN:output:0/Tree_Encoder/Batch_Norm_5_256/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2/
-Tree_Encoder/Batch_Norm_5_256/batchnorm/mul_1ó
8Tree_Encoder/Batch_Norm_5_256/batchnorm/ReadVariableOp_1ReadVariableOpAtree_encoder_batch_norm_5_256_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02:
8Tree_Encoder/Batch_Norm_5_256/batchnorm/ReadVariableOp_1þ
-Tree_Encoder/Batch_Norm_5_256/batchnorm/mul_2Mul@Tree_Encoder/Batch_Norm_5_256/batchnorm/ReadVariableOp_1:value:0/Tree_Encoder/Batch_Norm_5_256/batchnorm/mul:z:0*
T0*
_output_shapes	
:2/
-Tree_Encoder/Batch_Norm_5_256/batchnorm/mul_2ó
8Tree_Encoder/Batch_Norm_5_256/batchnorm/ReadVariableOp_2ReadVariableOpAtree_encoder_batch_norm_5_256_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02:
8Tree_Encoder/Batch_Norm_5_256/batchnorm/ReadVariableOp_2ü
+Tree_Encoder/Batch_Norm_5_256/batchnorm/subSub@Tree_Encoder/Batch_Norm_5_256/batchnorm/ReadVariableOp_2:value:01Tree_Encoder/Batch_Norm_5_256/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2-
+Tree_Encoder/Batch_Norm_5_256/batchnorm/sub
-Tree_Encoder/Batch_Norm_5_256/batchnorm/add_1AddV21Tree_Encoder/Batch_Norm_5_256/batchnorm/mul_1:z:0/Tree_Encoder/Batch_Norm_5_256/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2/
-Tree_Encoder/Batch_Norm_5_256/batchnorm/add_1í
6Tree_Encoder/Batch_Norm_3_256/batchnorm/ReadVariableOpReadVariableOp?tree_encoder_batch_norm_3_256_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype028
6Tree_Encoder/Batch_Norm_3_256/batchnorm/ReadVariableOp£
-Tree_Encoder/Batch_Norm_3_256/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2/
-Tree_Encoder/Batch_Norm_3_256/batchnorm/add/y
+Tree_Encoder/Batch_Norm_3_256/batchnorm/addAddV2>Tree_Encoder/Batch_Norm_3_256/batchnorm/ReadVariableOp:value:06Tree_Encoder/Batch_Norm_3_256/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2-
+Tree_Encoder/Batch_Norm_3_256/batchnorm/add¾
-Tree_Encoder/Batch_Norm_3_256/batchnorm/RsqrtRsqrt/Tree_Encoder/Batch_Norm_3_256/batchnorm/add:z:0*
T0*
_output_shapes	
:2/
-Tree_Encoder/Batch_Norm_3_256/batchnorm/Rsqrtù
:Tree_Encoder/Batch_Norm_3_256/batchnorm/mul/ReadVariableOpReadVariableOpCtree_encoder_batch_norm_3_256_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02<
:Tree_Encoder/Batch_Norm_3_256/batchnorm/mul/ReadVariableOpþ
+Tree_Encoder/Batch_Norm_3_256/batchnorm/mulMul1Tree_Encoder/Batch_Norm_3_256/batchnorm/Rsqrt:y:0BTree_Encoder/Batch_Norm_3_256/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2-
+Tree_Encoder/Batch_Norm_3_256/batchnorm/mul
-Tree_Encoder/Batch_Norm_3_256/batchnorm/mul_1Mul,Tree_Encoder/Conv1D_3_256/IdentityN:output:0/Tree_Encoder/Batch_Norm_3_256/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2/
-Tree_Encoder/Batch_Norm_3_256/batchnorm/mul_1ó
8Tree_Encoder/Batch_Norm_3_256/batchnorm/ReadVariableOp_1ReadVariableOpAtree_encoder_batch_norm_3_256_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02:
8Tree_Encoder/Batch_Norm_3_256/batchnorm/ReadVariableOp_1þ
-Tree_Encoder/Batch_Norm_3_256/batchnorm/mul_2Mul@Tree_Encoder/Batch_Norm_3_256/batchnorm/ReadVariableOp_1:value:0/Tree_Encoder/Batch_Norm_3_256/batchnorm/mul:z:0*
T0*
_output_shapes	
:2/
-Tree_Encoder/Batch_Norm_3_256/batchnorm/mul_2ó
8Tree_Encoder/Batch_Norm_3_256/batchnorm/ReadVariableOp_2ReadVariableOpAtree_encoder_batch_norm_3_256_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02:
8Tree_Encoder/Batch_Norm_3_256/batchnorm/ReadVariableOp_2ü
+Tree_Encoder/Batch_Norm_3_256/batchnorm/subSub@Tree_Encoder/Batch_Norm_3_256/batchnorm/ReadVariableOp_2:value:01Tree_Encoder/Batch_Norm_3_256/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2-
+Tree_Encoder/Batch_Norm_3_256/batchnorm/sub
-Tree_Encoder/Batch_Norm_3_256/batchnorm/add_1AddV21Tree_Encoder/Batch_Norm_3_256/batchnorm/mul_1:z:0/Tree_Encoder/Batch_Norm_3_256/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2/
-Tree_Encoder/Batch_Norm_3_256/batchnorm/add_1 
-Tree_Encoder/Max_Pool_3/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2/
-Tree_Encoder/Max_Pool_3/Max/reduction_indicesß
Tree_Encoder/Max_Pool_3/MaxMax1Tree_Encoder/Batch_Norm_3_256/batchnorm/add_1:z:06Tree_Encoder/Max_Pool_3/Max/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tree_Encoder/Max_Pool_3/Max 
-Tree_Encoder/Max_Pool_5/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2/
-Tree_Encoder/Max_Pool_5/Max/reduction_indicesß
Tree_Encoder/Max_Pool_5/MaxMax1Tree_Encoder/Batch_Norm_5_256/batchnorm/add_1:z:06Tree_Encoder/Max_Pool_5/Max/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tree_Encoder/Max_Pool_5/Max 
-Tree_Encoder/Max_Pool_7/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2/
-Tree_Encoder/Max_Pool_7/Max/reduction_indicesß
Tree_Encoder/Max_Pool_7/MaxMax1Tree_Encoder/Batch_Norm_7_256/batchnorm/add_1:z:06Tree_Encoder/Max_Pool_7/Max/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tree_Encoder/Max_Pool_7/Max
)Tree_Encoder/Pool_Concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2+
)Tree_Encoder/Pool_Concatenate/concat/axisº
$Tree_Encoder/Pool_Concatenate/concatConcatV2$Tree_Encoder/Max_Pool_3/Max:output:0$Tree_Encoder/Max_Pool_5/Max:output:0$Tree_Encoder/Max_Pool_7/Max:output:02Tree_Encoder/Pool_Concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$Tree_Encoder/Pool_Concatenate/concatÄ
)Tree_Encoder/Concatenate_Dropout/IdentityIdentity-Tree_Encoder/Pool_Concatenate/concat:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)Tree_Encoder/Concatenate_Dropout/IdentityÖ
-Tree_Encoder/Prediction/MatMul/ReadVariableOpReadVariableOp6tree_encoder_prediction_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02/
-Tree_Encoder/Prediction/MatMul/ReadVariableOpç
Tree_Encoder/Prediction/MatMulMatMul2Tree_Encoder/Concatenate_Dropout/Identity:output:05Tree_Encoder/Prediction/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
Tree_Encoder/Prediction/MatMulÔ
.Tree_Encoder/Prediction/BiasAdd/ReadVariableOpReadVariableOp7tree_encoder_prediction_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.Tree_Encoder/Prediction/BiasAdd/ReadVariableOpá
Tree_Encoder/Prediction/BiasAddBiasAdd(Tree_Encoder/Prediction/MatMul:product:06Tree_Encoder/Prediction/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
Tree_Encoder/Prediction/BiasAdd©
Tree_Encoder/Prediction/SigmoidSigmoid(Tree_Encoder/Prediction/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
Tree_Encoder/Prediction/Sigmoid
IdentityIdentity#Tree_Encoder/Prediction/Sigmoid:y:07^Tree_Encoder/Batch_Norm_3_128/batchnorm/ReadVariableOp9^Tree_Encoder/Batch_Norm_3_128/batchnorm/ReadVariableOp_19^Tree_Encoder/Batch_Norm_3_128/batchnorm/ReadVariableOp_2;^Tree_Encoder/Batch_Norm_3_128/batchnorm/mul/ReadVariableOp7^Tree_Encoder/Batch_Norm_3_256/batchnorm/ReadVariableOp9^Tree_Encoder/Batch_Norm_3_256/batchnorm/ReadVariableOp_19^Tree_Encoder/Batch_Norm_3_256/batchnorm/ReadVariableOp_2;^Tree_Encoder/Batch_Norm_3_256/batchnorm/mul/ReadVariableOp7^Tree_Encoder/Batch_Norm_5_128/batchnorm/ReadVariableOp9^Tree_Encoder/Batch_Norm_5_128/batchnorm/ReadVariableOp_19^Tree_Encoder/Batch_Norm_5_128/batchnorm/ReadVariableOp_2;^Tree_Encoder/Batch_Norm_5_128/batchnorm/mul/ReadVariableOp7^Tree_Encoder/Batch_Norm_5_256/batchnorm/ReadVariableOp9^Tree_Encoder/Batch_Norm_5_256/batchnorm/ReadVariableOp_19^Tree_Encoder/Batch_Norm_5_256/batchnorm/ReadVariableOp_2;^Tree_Encoder/Batch_Norm_5_256/batchnorm/mul/ReadVariableOp7^Tree_Encoder/Batch_Norm_7_128/batchnorm/ReadVariableOp9^Tree_Encoder/Batch_Norm_7_128/batchnorm/ReadVariableOp_19^Tree_Encoder/Batch_Norm_7_128/batchnorm/ReadVariableOp_2;^Tree_Encoder/Batch_Norm_7_128/batchnorm/mul/ReadVariableOp7^Tree_Encoder/Batch_Norm_7_256/batchnorm/ReadVariableOp9^Tree_Encoder/Batch_Norm_7_256/batchnorm/ReadVariableOp_19^Tree_Encoder/Batch_Norm_7_256/batchnorm/ReadVariableOp_2;^Tree_Encoder/Batch_Norm_7_256/batchnorm/mul/ReadVariableOp1^Tree_Encoder/Conv1D_3_128/BiasAdd/ReadVariableOp=^Tree_Encoder/Conv1D_3_128/conv1d/ExpandDims_1/ReadVariableOp1^Tree_Encoder/Conv1D_3_256/BiasAdd/ReadVariableOp=^Tree_Encoder/Conv1D_3_256/conv1d/ExpandDims_1/ReadVariableOp1^Tree_Encoder/Conv1D_5_128/BiasAdd/ReadVariableOp=^Tree_Encoder/Conv1D_5_128/conv1d/ExpandDims_1/ReadVariableOp1^Tree_Encoder/Conv1D_5_256/BiasAdd/ReadVariableOp=^Tree_Encoder/Conv1D_5_256/conv1d/ExpandDims_1/ReadVariableOp1^Tree_Encoder/Conv1D_7_128/BiasAdd/ReadVariableOp=^Tree_Encoder/Conv1D_7_128/conv1d/ExpandDims_1/ReadVariableOp1^Tree_Encoder/Conv1D_7_256/BiasAdd/ReadVariableOp=^Tree_Encoder/Conv1D_7_256/conv1d/ExpandDims_1/ReadVariableOp/^Tree_Encoder/Prediction/BiasAdd/ReadVariableOp.^Tree_Encoder/Prediction/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapeso
m:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2p
6Tree_Encoder/Batch_Norm_3_128/batchnorm/ReadVariableOp6Tree_Encoder/Batch_Norm_3_128/batchnorm/ReadVariableOp2t
8Tree_Encoder/Batch_Norm_3_128/batchnorm/ReadVariableOp_18Tree_Encoder/Batch_Norm_3_128/batchnorm/ReadVariableOp_12t
8Tree_Encoder/Batch_Norm_3_128/batchnorm/ReadVariableOp_28Tree_Encoder/Batch_Norm_3_128/batchnorm/ReadVariableOp_22x
:Tree_Encoder/Batch_Norm_3_128/batchnorm/mul/ReadVariableOp:Tree_Encoder/Batch_Norm_3_128/batchnorm/mul/ReadVariableOp2p
6Tree_Encoder/Batch_Norm_3_256/batchnorm/ReadVariableOp6Tree_Encoder/Batch_Norm_3_256/batchnorm/ReadVariableOp2t
8Tree_Encoder/Batch_Norm_3_256/batchnorm/ReadVariableOp_18Tree_Encoder/Batch_Norm_3_256/batchnorm/ReadVariableOp_12t
8Tree_Encoder/Batch_Norm_3_256/batchnorm/ReadVariableOp_28Tree_Encoder/Batch_Norm_3_256/batchnorm/ReadVariableOp_22x
:Tree_Encoder/Batch_Norm_3_256/batchnorm/mul/ReadVariableOp:Tree_Encoder/Batch_Norm_3_256/batchnorm/mul/ReadVariableOp2p
6Tree_Encoder/Batch_Norm_5_128/batchnorm/ReadVariableOp6Tree_Encoder/Batch_Norm_5_128/batchnorm/ReadVariableOp2t
8Tree_Encoder/Batch_Norm_5_128/batchnorm/ReadVariableOp_18Tree_Encoder/Batch_Norm_5_128/batchnorm/ReadVariableOp_12t
8Tree_Encoder/Batch_Norm_5_128/batchnorm/ReadVariableOp_28Tree_Encoder/Batch_Norm_5_128/batchnorm/ReadVariableOp_22x
:Tree_Encoder/Batch_Norm_5_128/batchnorm/mul/ReadVariableOp:Tree_Encoder/Batch_Norm_5_128/batchnorm/mul/ReadVariableOp2p
6Tree_Encoder/Batch_Norm_5_256/batchnorm/ReadVariableOp6Tree_Encoder/Batch_Norm_5_256/batchnorm/ReadVariableOp2t
8Tree_Encoder/Batch_Norm_5_256/batchnorm/ReadVariableOp_18Tree_Encoder/Batch_Norm_5_256/batchnorm/ReadVariableOp_12t
8Tree_Encoder/Batch_Norm_5_256/batchnorm/ReadVariableOp_28Tree_Encoder/Batch_Norm_5_256/batchnorm/ReadVariableOp_22x
:Tree_Encoder/Batch_Norm_5_256/batchnorm/mul/ReadVariableOp:Tree_Encoder/Batch_Norm_5_256/batchnorm/mul/ReadVariableOp2p
6Tree_Encoder/Batch_Norm_7_128/batchnorm/ReadVariableOp6Tree_Encoder/Batch_Norm_7_128/batchnorm/ReadVariableOp2t
8Tree_Encoder/Batch_Norm_7_128/batchnorm/ReadVariableOp_18Tree_Encoder/Batch_Norm_7_128/batchnorm/ReadVariableOp_12t
8Tree_Encoder/Batch_Norm_7_128/batchnorm/ReadVariableOp_28Tree_Encoder/Batch_Norm_7_128/batchnorm/ReadVariableOp_22x
:Tree_Encoder/Batch_Norm_7_128/batchnorm/mul/ReadVariableOp:Tree_Encoder/Batch_Norm_7_128/batchnorm/mul/ReadVariableOp2p
6Tree_Encoder/Batch_Norm_7_256/batchnorm/ReadVariableOp6Tree_Encoder/Batch_Norm_7_256/batchnorm/ReadVariableOp2t
8Tree_Encoder/Batch_Norm_7_256/batchnorm/ReadVariableOp_18Tree_Encoder/Batch_Norm_7_256/batchnorm/ReadVariableOp_12t
8Tree_Encoder/Batch_Norm_7_256/batchnorm/ReadVariableOp_28Tree_Encoder/Batch_Norm_7_256/batchnorm/ReadVariableOp_22x
:Tree_Encoder/Batch_Norm_7_256/batchnorm/mul/ReadVariableOp:Tree_Encoder/Batch_Norm_7_256/batchnorm/mul/ReadVariableOp2d
0Tree_Encoder/Conv1D_3_128/BiasAdd/ReadVariableOp0Tree_Encoder/Conv1D_3_128/BiasAdd/ReadVariableOp2|
<Tree_Encoder/Conv1D_3_128/conv1d/ExpandDims_1/ReadVariableOp<Tree_Encoder/Conv1D_3_128/conv1d/ExpandDims_1/ReadVariableOp2d
0Tree_Encoder/Conv1D_3_256/BiasAdd/ReadVariableOp0Tree_Encoder/Conv1D_3_256/BiasAdd/ReadVariableOp2|
<Tree_Encoder/Conv1D_3_256/conv1d/ExpandDims_1/ReadVariableOp<Tree_Encoder/Conv1D_3_256/conv1d/ExpandDims_1/ReadVariableOp2d
0Tree_Encoder/Conv1D_5_128/BiasAdd/ReadVariableOp0Tree_Encoder/Conv1D_5_128/BiasAdd/ReadVariableOp2|
<Tree_Encoder/Conv1D_5_128/conv1d/ExpandDims_1/ReadVariableOp<Tree_Encoder/Conv1D_5_128/conv1d/ExpandDims_1/ReadVariableOp2d
0Tree_Encoder/Conv1D_5_256/BiasAdd/ReadVariableOp0Tree_Encoder/Conv1D_5_256/BiasAdd/ReadVariableOp2|
<Tree_Encoder/Conv1D_5_256/conv1d/ExpandDims_1/ReadVariableOp<Tree_Encoder/Conv1D_5_256/conv1d/ExpandDims_1/ReadVariableOp2d
0Tree_Encoder/Conv1D_7_128/BiasAdd/ReadVariableOp0Tree_Encoder/Conv1D_7_128/BiasAdd/ReadVariableOp2|
<Tree_Encoder/Conv1D_7_128/conv1d/ExpandDims_1/ReadVariableOp<Tree_Encoder/Conv1D_7_128/conv1d/ExpandDims_1/ReadVariableOp2d
0Tree_Encoder/Conv1D_7_256/BiasAdd/ReadVariableOp0Tree_Encoder/Conv1D_7_256/BiasAdd/ReadVariableOp2|
<Tree_Encoder/Conv1D_7_256/conv1d/ExpandDims_1/ReadVariableOp<Tree_Encoder/Conv1D_7_256/conv1d/ExpandDims_1/ReadVariableOp2`
.Tree_Encoder/Prediction/BiasAdd/ReadVariableOp.Tree_Encoder/Prediction/BiasAdd/ReadVariableOp2^
-Tree_Encoder/Prediction/MatMul/ReadVariableOp-Tree_Encoder/Prediction/MatMul/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameInputs
¾
Ô
5__inference_Batch_Norm_7_256_layer_call_fn_1916624947

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_Batch_Norm_7_256_layer_call_and_return_conditional_losses_19166223582
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ì
	
(__inference_signature_wrapper_1916623629

inputs
unknown:
	unknown_0:	!
	unknown_1:
	unknown_2:	!
	unknown_3:
	unknown_4:	
	unknown_5:	
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:	

unknown_11:	

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	"

unknown_17:

unknown_18:	"

unknown_19:

unknown_20:	"

unknown_21:

unknown_22:	

unknown_23:	

unknown_24:	

unknown_25:	

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:	

unknown_30:	

unknown_31:	

unknown_32:	

unknown_33:	

unknown_34:	

unknown_35:	

unknown_36:
identity¢StatefulPartitionedCallÇ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8 *.
f)R'
%__inference__wrapped_model_19166215242
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapeso
m:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameInputs
ü*
í
P__inference_Batch_Norm_7_256_layer_call_and_return_conditional_losses_1916622418

inputs6
'assignmovingavg_readvariableop_resource:	8
)assignmovingavg_1_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	0
!batchnorm_readvariableop_resource:	
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:2
moments/StopGradient²
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices·
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay¥
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg/mul¿
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay«
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp¡
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü
 
L__inference_Conv1D_3_128_layer_call_and_return_conditional_losses_1916624384

inputsC
+conv1d_expanddims_1_readvariableop_resource:.
biasadd_readvariableop_resource:	

identity_1¢BiasAdd/ReadVariableOp¢"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims/dim 
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
conv1d/ExpandDimsº
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim¹
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv1d/ExpandDims_1À
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAddo
SigmoidSigmoidBiasAdd:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
Sigmoidp
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
muli
IdentityIdentitymul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

IdentityÕ
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1916624377*V
_output_shapesD
B:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	IdentityN¶

Identity_1IdentityIdentityN:output:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Â
r
S__inference_Concatenate_Dropout_layer_call_and_return_conditional_losses_1916622912

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeµ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
dropout/GreaterEqual/y¿
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¸

ü
J__inference_Prediction_layer_call_and_return_conditional_losses_1916622796

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoid
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ
³
P__inference_Batch_Norm_3_256_layer_call_and_return_conditional_losses_1916624820

inputs0
!batchnorm_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	2
#batchnorm_readvariableop_1_resource:	2
#batchnorm_readvariableop_2_resource:	
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1é
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü
 
L__inference_Conv1D_7_128_layer_call_and_return_conditional_losses_1916624444

inputsC
+conv1d_expanddims_1_readvariableop_resource:.
biasadd_readvariableop_resource:	

identity_1¢BiasAdd/ReadVariableOp¢"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims/dim 
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
conv1d/ExpandDimsº
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim¹
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv1d/ExpandDims_1À
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAddo
SigmoidSigmoidBiasAdd:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
Sigmoidp
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
muli
IdentityIdentitymul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

IdentityÕ
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1916624437*V
_output_shapesD
B:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	IdentityN¶

Identity_1IdentityIdentityN:output:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
î
'
L__inference_Tree_Encoder_layer_call_and_return_conditional_losses_1916624327

inputsP
8conv1d_7_128_conv1d_expanddims_1_readvariableop_resource:;
,conv1d_7_128_biasadd_readvariableop_resource:	P
8conv1d_5_128_conv1d_expanddims_1_readvariableop_resource:;
,conv1d_5_128_biasadd_readvariableop_resource:	P
8conv1d_3_128_conv1d_expanddims_1_readvariableop_resource:;
,conv1d_3_128_biasadd_readvariableop_resource:	G
8batch_norm_7_128_assignmovingavg_readvariableop_resource:	I
:batch_norm_7_128_assignmovingavg_1_readvariableop_resource:	E
6batch_norm_7_128_batchnorm_mul_readvariableop_resource:	A
2batch_norm_7_128_batchnorm_readvariableop_resource:	G
8batch_norm_5_128_assignmovingavg_readvariableop_resource:	I
:batch_norm_5_128_assignmovingavg_1_readvariableop_resource:	E
6batch_norm_5_128_batchnorm_mul_readvariableop_resource:	A
2batch_norm_5_128_batchnorm_readvariableop_resource:	G
8batch_norm_3_128_assignmovingavg_readvariableop_resource:	I
:batch_norm_3_128_assignmovingavg_1_readvariableop_resource:	E
6batch_norm_3_128_batchnorm_mul_readvariableop_resource:	A
2batch_norm_3_128_batchnorm_readvariableop_resource:	P
8conv1d_7_256_conv1d_expanddims_1_readvariableop_resource:;
,conv1d_7_256_biasadd_readvariableop_resource:	P
8conv1d_5_256_conv1d_expanddims_1_readvariableop_resource:;
,conv1d_5_256_biasadd_readvariableop_resource:	P
8conv1d_3_256_conv1d_expanddims_1_readvariableop_resource:;
,conv1d_3_256_biasadd_readvariableop_resource:	G
8batch_norm_7_256_assignmovingavg_readvariableop_resource:	I
:batch_norm_7_256_assignmovingavg_1_readvariableop_resource:	E
6batch_norm_7_256_batchnorm_mul_readvariableop_resource:	A
2batch_norm_7_256_batchnorm_readvariableop_resource:	G
8batch_norm_5_256_assignmovingavg_readvariableop_resource:	I
:batch_norm_5_256_assignmovingavg_1_readvariableop_resource:	E
6batch_norm_5_256_batchnorm_mul_readvariableop_resource:	A
2batch_norm_5_256_batchnorm_readvariableop_resource:	G
8batch_norm_3_256_assignmovingavg_readvariableop_resource:	I
:batch_norm_3_256_assignmovingavg_1_readvariableop_resource:	E
6batch_norm_3_256_batchnorm_mul_readvariableop_resource:	A
2batch_norm_3_256_batchnorm_readvariableop_resource:	<
)prediction_matmul_readvariableop_resource:	8
*prediction_biasadd_readvariableop_resource:
identity¢ Batch_Norm_3_128/AssignMovingAvg¢/Batch_Norm_3_128/AssignMovingAvg/ReadVariableOp¢"Batch_Norm_3_128/AssignMovingAvg_1¢1Batch_Norm_3_128/AssignMovingAvg_1/ReadVariableOp¢)Batch_Norm_3_128/batchnorm/ReadVariableOp¢-Batch_Norm_3_128/batchnorm/mul/ReadVariableOp¢ Batch_Norm_3_256/AssignMovingAvg¢/Batch_Norm_3_256/AssignMovingAvg/ReadVariableOp¢"Batch_Norm_3_256/AssignMovingAvg_1¢1Batch_Norm_3_256/AssignMovingAvg_1/ReadVariableOp¢)Batch_Norm_3_256/batchnorm/ReadVariableOp¢-Batch_Norm_3_256/batchnorm/mul/ReadVariableOp¢ Batch_Norm_5_128/AssignMovingAvg¢/Batch_Norm_5_128/AssignMovingAvg/ReadVariableOp¢"Batch_Norm_5_128/AssignMovingAvg_1¢1Batch_Norm_5_128/AssignMovingAvg_1/ReadVariableOp¢)Batch_Norm_5_128/batchnorm/ReadVariableOp¢-Batch_Norm_5_128/batchnorm/mul/ReadVariableOp¢ Batch_Norm_5_256/AssignMovingAvg¢/Batch_Norm_5_256/AssignMovingAvg/ReadVariableOp¢"Batch_Norm_5_256/AssignMovingAvg_1¢1Batch_Norm_5_256/AssignMovingAvg_1/ReadVariableOp¢)Batch_Norm_5_256/batchnorm/ReadVariableOp¢-Batch_Norm_5_256/batchnorm/mul/ReadVariableOp¢ Batch_Norm_7_128/AssignMovingAvg¢/Batch_Norm_7_128/AssignMovingAvg/ReadVariableOp¢"Batch_Norm_7_128/AssignMovingAvg_1¢1Batch_Norm_7_128/AssignMovingAvg_1/ReadVariableOp¢)Batch_Norm_7_128/batchnorm/ReadVariableOp¢-Batch_Norm_7_128/batchnorm/mul/ReadVariableOp¢ Batch_Norm_7_256/AssignMovingAvg¢/Batch_Norm_7_256/AssignMovingAvg/ReadVariableOp¢"Batch_Norm_7_256/AssignMovingAvg_1¢1Batch_Norm_7_256/AssignMovingAvg_1/ReadVariableOp¢)Batch_Norm_7_256/batchnorm/ReadVariableOp¢-Batch_Norm_7_256/batchnorm/mul/ReadVariableOp¢#Conv1D_3_128/BiasAdd/ReadVariableOp¢/Conv1D_3_128/conv1d/ExpandDims_1/ReadVariableOp¢#Conv1D_3_256/BiasAdd/ReadVariableOp¢/Conv1D_3_256/conv1d/ExpandDims_1/ReadVariableOp¢#Conv1D_5_128/BiasAdd/ReadVariableOp¢/Conv1D_5_128/conv1d/ExpandDims_1/ReadVariableOp¢#Conv1D_5_256/BiasAdd/ReadVariableOp¢/Conv1D_5_256/conv1d/ExpandDims_1/ReadVariableOp¢#Conv1D_7_128/BiasAdd/ReadVariableOp¢/Conv1D_7_128/conv1d/ExpandDims_1/ReadVariableOp¢#Conv1D_7_256/BiasAdd/ReadVariableOp¢/Conv1D_7_256/conv1d/ExpandDims_1/ReadVariableOp¢!Prediction/BiasAdd/ReadVariableOp¢ Prediction/MatMul/ReadVariableOp
Embedding_Dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2!
Embedding_Dropout/dropout/Const·
Embedding_Dropout/dropout/MulMulinputs(Embedding_Dropout/dropout/Const:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Embedding_Dropout/dropout/Mulx
Embedding_Dropout/dropout/ShapeShapeinputs*
T0*
_output_shapes
:2!
Embedding_Dropout/dropout/Shapeø
6Embedding_Dropout/dropout/random_uniform/RandomUniformRandomUniform(Embedding_Dropout/dropout/Shape:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype028
6Embedding_Dropout/dropout/random_uniform/RandomUniform
(Embedding_Dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2*
(Embedding_Dropout/dropout/GreaterEqual/y
&Embedding_Dropout/dropout/GreaterEqualGreaterEqual?Embedding_Dropout/dropout/random_uniform/RandomUniform:output:01Embedding_Dropout/dropout/GreaterEqual/y:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2(
&Embedding_Dropout/dropout/GreaterEqualÃ
Embedding_Dropout/dropout/CastCast*Embedding_Dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2 
Embedding_Dropout/dropout/CastÐ
Embedding_Dropout/dropout/Mul_1Mul!Embedding_Dropout/dropout/Mul:z:0"Embedding_Dropout/dropout/Cast:y:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2!
Embedding_Dropout/dropout/Mul_1
"Conv1D_7_128/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2$
"Conv1D_7_128/conv1d/ExpandDims/dimä
Conv1D_7_128/conv1d/ExpandDims
ExpandDims#Embedding_Dropout/dropout/Mul_1:z:0+Conv1D_7_128/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2 
Conv1D_7_128/conv1d/ExpandDimsá
/Conv1D_7_128/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp8conv1d_7_128_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype021
/Conv1D_7_128/conv1d/ExpandDims_1/ReadVariableOp
$Conv1D_7_128/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Conv1D_7_128/conv1d/ExpandDims_1/dimí
 Conv1D_7_128/conv1d/ExpandDims_1
ExpandDims7Conv1D_7_128/conv1d/ExpandDims_1/ReadVariableOp:value:0-Conv1D_7_128/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2"
 Conv1D_7_128/conv1d/ExpandDims_1ô
Conv1D_7_128/conv1dConv2D'Conv1D_7_128/conv1d/ExpandDims:output:0)Conv1D_7_128/conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv1D_7_128/conv1dÃ
Conv1D_7_128/conv1d/SqueezeSqueezeConv1D_7_128/conv1d:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
Conv1D_7_128/conv1d/Squeeze´
#Conv1D_7_128/BiasAdd/ReadVariableOpReadVariableOp,conv1d_7_128_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#Conv1D_7_128/BiasAdd/ReadVariableOpÊ
Conv1D_7_128/BiasAddBiasAdd$Conv1D_7_128/conv1d/Squeeze:output:0+Conv1D_7_128/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_7_128/BiasAdd
Conv1D_7_128/SigmoidSigmoidConv1D_7_128/BiasAdd:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_7_128/Sigmoid¤
Conv1D_7_128/mulMulConv1D_7_128/BiasAdd:output:0Conv1D_7_128/Sigmoid:y:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_7_128/mul
Conv1D_7_128/IdentityIdentityConv1D_7_128/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_7_128/Identity
Conv1D_7_128/IdentityN	IdentityNConv1D_7_128/mul:z:0Conv1D_7_128/BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1916624032*V
_output_shapesD
B:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_7_128/IdentityN
"Conv1D_5_128/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2$
"Conv1D_5_128/conv1d/ExpandDims/dimä
Conv1D_5_128/conv1d/ExpandDims
ExpandDims#Embedding_Dropout/dropout/Mul_1:z:0+Conv1D_5_128/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2 
Conv1D_5_128/conv1d/ExpandDimsá
/Conv1D_5_128/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp8conv1d_5_128_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype021
/Conv1D_5_128/conv1d/ExpandDims_1/ReadVariableOp
$Conv1D_5_128/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Conv1D_5_128/conv1d/ExpandDims_1/dimí
 Conv1D_5_128/conv1d/ExpandDims_1
ExpandDims7Conv1D_5_128/conv1d/ExpandDims_1/ReadVariableOp:value:0-Conv1D_5_128/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2"
 Conv1D_5_128/conv1d/ExpandDims_1ô
Conv1D_5_128/conv1dConv2D'Conv1D_5_128/conv1d/ExpandDims:output:0)Conv1D_5_128/conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv1D_5_128/conv1dÃ
Conv1D_5_128/conv1d/SqueezeSqueezeConv1D_5_128/conv1d:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
Conv1D_5_128/conv1d/Squeeze´
#Conv1D_5_128/BiasAdd/ReadVariableOpReadVariableOp,conv1d_5_128_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#Conv1D_5_128/BiasAdd/ReadVariableOpÊ
Conv1D_5_128/BiasAddBiasAdd$Conv1D_5_128/conv1d/Squeeze:output:0+Conv1D_5_128/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_5_128/BiasAdd
Conv1D_5_128/SigmoidSigmoidConv1D_5_128/BiasAdd:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_5_128/Sigmoid¤
Conv1D_5_128/mulMulConv1D_5_128/BiasAdd:output:0Conv1D_5_128/Sigmoid:y:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_5_128/mul
Conv1D_5_128/IdentityIdentityConv1D_5_128/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_5_128/Identity
Conv1D_5_128/IdentityN	IdentityNConv1D_5_128/mul:z:0Conv1D_5_128/BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1916624049*V
_output_shapesD
B:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_5_128/IdentityN
"Conv1D_3_128/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2$
"Conv1D_3_128/conv1d/ExpandDims/dimä
Conv1D_3_128/conv1d/ExpandDims
ExpandDims#Embedding_Dropout/dropout/Mul_1:z:0+Conv1D_3_128/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2 
Conv1D_3_128/conv1d/ExpandDimsá
/Conv1D_3_128/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp8conv1d_3_128_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype021
/Conv1D_3_128/conv1d/ExpandDims_1/ReadVariableOp
$Conv1D_3_128/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Conv1D_3_128/conv1d/ExpandDims_1/dimí
 Conv1D_3_128/conv1d/ExpandDims_1
ExpandDims7Conv1D_3_128/conv1d/ExpandDims_1/ReadVariableOp:value:0-Conv1D_3_128/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2"
 Conv1D_3_128/conv1d/ExpandDims_1ô
Conv1D_3_128/conv1dConv2D'Conv1D_3_128/conv1d/ExpandDims:output:0)Conv1D_3_128/conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv1D_3_128/conv1dÃ
Conv1D_3_128/conv1d/SqueezeSqueezeConv1D_3_128/conv1d:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
Conv1D_3_128/conv1d/Squeeze´
#Conv1D_3_128/BiasAdd/ReadVariableOpReadVariableOp,conv1d_3_128_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#Conv1D_3_128/BiasAdd/ReadVariableOpÊ
Conv1D_3_128/BiasAddBiasAdd$Conv1D_3_128/conv1d/Squeeze:output:0+Conv1D_3_128/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_3_128/BiasAdd
Conv1D_3_128/SigmoidSigmoidConv1D_3_128/BiasAdd:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_3_128/Sigmoid¤
Conv1D_3_128/mulMulConv1D_3_128/BiasAdd:output:0Conv1D_3_128/Sigmoid:y:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_3_128/mul
Conv1D_3_128/IdentityIdentityConv1D_3_128/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_3_128/Identity
Conv1D_3_128/IdentityN	IdentityNConv1D_3_128/mul:z:0Conv1D_3_128/BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1916624066*V
_output_shapesD
B:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_3_128/IdentityN³
/Batch_Norm_7_128/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       21
/Batch_Norm_7_128/moments/mean/reduction_indicesà
Batch_Norm_7_128/moments/meanMeanConv1D_7_128/IdentityN:output:08Batch_Norm_7_128/moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:*
	keep_dims(2
Batch_Norm_7_128/moments/mean´
%Batch_Norm_7_128/moments/StopGradientStopGradient&Batch_Norm_7_128/moments/mean:output:0*
T0*#
_output_shapes
:2'
%Batch_Norm_7_128/moments/StopGradientþ
*Batch_Norm_7_128/moments/SquaredDifferenceSquaredDifferenceConv1D_7_128/IdentityN:output:0.Batch_Norm_7_128/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2,
*Batch_Norm_7_128/moments/SquaredDifference»
3Batch_Norm_7_128/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       25
3Batch_Norm_7_128/moments/variance/reduction_indicesû
!Batch_Norm_7_128/moments/varianceMean.Batch_Norm_7_128/moments/SquaredDifference:z:0<Batch_Norm_7_128/moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:*
	keep_dims(2#
!Batch_Norm_7_128/moments/varianceµ
 Batch_Norm_7_128/moments/SqueezeSqueeze&Batch_Norm_7_128/moments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2"
 Batch_Norm_7_128/moments/Squeeze½
"Batch_Norm_7_128/moments/Squeeze_1Squeeze*Batch_Norm_7_128/moments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2$
"Batch_Norm_7_128/moments/Squeeze_1
&Batch_Norm_7_128/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2(
&Batch_Norm_7_128/AssignMovingAvg/decayØ
/Batch_Norm_7_128/AssignMovingAvg/ReadVariableOpReadVariableOp8batch_norm_7_128_assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype021
/Batch_Norm_7_128/AssignMovingAvg/ReadVariableOpÝ
$Batch_Norm_7_128/AssignMovingAvg/subSub7Batch_Norm_7_128/AssignMovingAvg/ReadVariableOp:value:0)Batch_Norm_7_128/moments/Squeeze:output:0*
T0*
_output_shapes	
:2&
$Batch_Norm_7_128/AssignMovingAvg/subÔ
$Batch_Norm_7_128/AssignMovingAvg/mulMul(Batch_Norm_7_128/AssignMovingAvg/sub:z:0/Batch_Norm_7_128/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:2&
$Batch_Norm_7_128/AssignMovingAvg/mul
 Batch_Norm_7_128/AssignMovingAvgAssignSubVariableOp8batch_norm_7_128_assignmovingavg_readvariableop_resource(Batch_Norm_7_128/AssignMovingAvg/mul:z:00^Batch_Norm_7_128/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02"
 Batch_Norm_7_128/AssignMovingAvg
(Batch_Norm_7_128/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2*
(Batch_Norm_7_128/AssignMovingAvg_1/decayÞ
1Batch_Norm_7_128/AssignMovingAvg_1/ReadVariableOpReadVariableOp:batch_norm_7_128_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype023
1Batch_Norm_7_128/AssignMovingAvg_1/ReadVariableOpå
&Batch_Norm_7_128/AssignMovingAvg_1/subSub9Batch_Norm_7_128/AssignMovingAvg_1/ReadVariableOp:value:0+Batch_Norm_7_128/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:2(
&Batch_Norm_7_128/AssignMovingAvg_1/subÜ
&Batch_Norm_7_128/AssignMovingAvg_1/mulMul*Batch_Norm_7_128/AssignMovingAvg_1/sub:z:01Batch_Norm_7_128/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:2(
&Batch_Norm_7_128/AssignMovingAvg_1/mul
"Batch_Norm_7_128/AssignMovingAvg_1AssignSubVariableOp:batch_norm_7_128_assignmovingavg_1_readvariableop_resource*Batch_Norm_7_128/AssignMovingAvg_1/mul:z:02^Batch_Norm_7_128/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02$
"Batch_Norm_7_128/AssignMovingAvg_1
 Batch_Norm_7_128/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2"
 Batch_Norm_7_128/batchnorm/add/yÇ
Batch_Norm_7_128/batchnorm/addAddV2+Batch_Norm_7_128/moments/Squeeze_1:output:0)Batch_Norm_7_128/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2 
Batch_Norm_7_128/batchnorm/add
 Batch_Norm_7_128/batchnorm/RsqrtRsqrt"Batch_Norm_7_128/batchnorm/add:z:0*
T0*
_output_shapes	
:2"
 Batch_Norm_7_128/batchnorm/RsqrtÒ
-Batch_Norm_7_128/batchnorm/mul/ReadVariableOpReadVariableOp6batch_norm_7_128_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02/
-Batch_Norm_7_128/batchnorm/mul/ReadVariableOpÊ
Batch_Norm_7_128/batchnorm/mulMul$Batch_Norm_7_128/batchnorm/Rsqrt:y:05Batch_Norm_7_128/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2 
Batch_Norm_7_128/batchnorm/mulÐ
 Batch_Norm_7_128/batchnorm/mul_1MulConv1D_7_128/IdentityN:output:0"Batch_Norm_7_128/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2"
 Batch_Norm_7_128/batchnorm/mul_1À
 Batch_Norm_7_128/batchnorm/mul_2Mul)Batch_Norm_7_128/moments/Squeeze:output:0"Batch_Norm_7_128/batchnorm/mul:z:0*
T0*
_output_shapes	
:2"
 Batch_Norm_7_128/batchnorm/mul_2Æ
)Batch_Norm_7_128/batchnorm/ReadVariableOpReadVariableOp2batch_norm_7_128_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02+
)Batch_Norm_7_128/batchnorm/ReadVariableOpÆ
Batch_Norm_7_128/batchnorm/subSub1Batch_Norm_7_128/batchnorm/ReadVariableOp:value:0$Batch_Norm_7_128/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2 
Batch_Norm_7_128/batchnorm/sub×
 Batch_Norm_7_128/batchnorm/add_1AddV2$Batch_Norm_7_128/batchnorm/mul_1:z:0"Batch_Norm_7_128/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2"
 Batch_Norm_7_128/batchnorm/add_1³
/Batch_Norm_5_128/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       21
/Batch_Norm_5_128/moments/mean/reduction_indicesà
Batch_Norm_5_128/moments/meanMeanConv1D_5_128/IdentityN:output:08Batch_Norm_5_128/moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:*
	keep_dims(2
Batch_Norm_5_128/moments/mean´
%Batch_Norm_5_128/moments/StopGradientStopGradient&Batch_Norm_5_128/moments/mean:output:0*
T0*#
_output_shapes
:2'
%Batch_Norm_5_128/moments/StopGradientþ
*Batch_Norm_5_128/moments/SquaredDifferenceSquaredDifferenceConv1D_5_128/IdentityN:output:0.Batch_Norm_5_128/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2,
*Batch_Norm_5_128/moments/SquaredDifference»
3Batch_Norm_5_128/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       25
3Batch_Norm_5_128/moments/variance/reduction_indicesû
!Batch_Norm_5_128/moments/varianceMean.Batch_Norm_5_128/moments/SquaredDifference:z:0<Batch_Norm_5_128/moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:*
	keep_dims(2#
!Batch_Norm_5_128/moments/varianceµ
 Batch_Norm_5_128/moments/SqueezeSqueeze&Batch_Norm_5_128/moments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2"
 Batch_Norm_5_128/moments/Squeeze½
"Batch_Norm_5_128/moments/Squeeze_1Squeeze*Batch_Norm_5_128/moments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2$
"Batch_Norm_5_128/moments/Squeeze_1
&Batch_Norm_5_128/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2(
&Batch_Norm_5_128/AssignMovingAvg/decayØ
/Batch_Norm_5_128/AssignMovingAvg/ReadVariableOpReadVariableOp8batch_norm_5_128_assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype021
/Batch_Norm_5_128/AssignMovingAvg/ReadVariableOpÝ
$Batch_Norm_5_128/AssignMovingAvg/subSub7Batch_Norm_5_128/AssignMovingAvg/ReadVariableOp:value:0)Batch_Norm_5_128/moments/Squeeze:output:0*
T0*
_output_shapes	
:2&
$Batch_Norm_5_128/AssignMovingAvg/subÔ
$Batch_Norm_5_128/AssignMovingAvg/mulMul(Batch_Norm_5_128/AssignMovingAvg/sub:z:0/Batch_Norm_5_128/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:2&
$Batch_Norm_5_128/AssignMovingAvg/mul
 Batch_Norm_5_128/AssignMovingAvgAssignSubVariableOp8batch_norm_5_128_assignmovingavg_readvariableop_resource(Batch_Norm_5_128/AssignMovingAvg/mul:z:00^Batch_Norm_5_128/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02"
 Batch_Norm_5_128/AssignMovingAvg
(Batch_Norm_5_128/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2*
(Batch_Norm_5_128/AssignMovingAvg_1/decayÞ
1Batch_Norm_5_128/AssignMovingAvg_1/ReadVariableOpReadVariableOp:batch_norm_5_128_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype023
1Batch_Norm_5_128/AssignMovingAvg_1/ReadVariableOpå
&Batch_Norm_5_128/AssignMovingAvg_1/subSub9Batch_Norm_5_128/AssignMovingAvg_1/ReadVariableOp:value:0+Batch_Norm_5_128/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:2(
&Batch_Norm_5_128/AssignMovingAvg_1/subÜ
&Batch_Norm_5_128/AssignMovingAvg_1/mulMul*Batch_Norm_5_128/AssignMovingAvg_1/sub:z:01Batch_Norm_5_128/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:2(
&Batch_Norm_5_128/AssignMovingAvg_1/mul
"Batch_Norm_5_128/AssignMovingAvg_1AssignSubVariableOp:batch_norm_5_128_assignmovingavg_1_readvariableop_resource*Batch_Norm_5_128/AssignMovingAvg_1/mul:z:02^Batch_Norm_5_128/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02$
"Batch_Norm_5_128/AssignMovingAvg_1
 Batch_Norm_5_128/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2"
 Batch_Norm_5_128/batchnorm/add/yÇ
Batch_Norm_5_128/batchnorm/addAddV2+Batch_Norm_5_128/moments/Squeeze_1:output:0)Batch_Norm_5_128/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2 
Batch_Norm_5_128/batchnorm/add
 Batch_Norm_5_128/batchnorm/RsqrtRsqrt"Batch_Norm_5_128/batchnorm/add:z:0*
T0*
_output_shapes	
:2"
 Batch_Norm_5_128/batchnorm/RsqrtÒ
-Batch_Norm_5_128/batchnorm/mul/ReadVariableOpReadVariableOp6batch_norm_5_128_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02/
-Batch_Norm_5_128/batchnorm/mul/ReadVariableOpÊ
Batch_Norm_5_128/batchnorm/mulMul$Batch_Norm_5_128/batchnorm/Rsqrt:y:05Batch_Norm_5_128/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2 
Batch_Norm_5_128/batchnorm/mulÐ
 Batch_Norm_5_128/batchnorm/mul_1MulConv1D_5_128/IdentityN:output:0"Batch_Norm_5_128/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2"
 Batch_Norm_5_128/batchnorm/mul_1À
 Batch_Norm_5_128/batchnorm/mul_2Mul)Batch_Norm_5_128/moments/Squeeze:output:0"Batch_Norm_5_128/batchnorm/mul:z:0*
T0*
_output_shapes	
:2"
 Batch_Norm_5_128/batchnorm/mul_2Æ
)Batch_Norm_5_128/batchnorm/ReadVariableOpReadVariableOp2batch_norm_5_128_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02+
)Batch_Norm_5_128/batchnorm/ReadVariableOpÆ
Batch_Norm_5_128/batchnorm/subSub1Batch_Norm_5_128/batchnorm/ReadVariableOp:value:0$Batch_Norm_5_128/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2 
Batch_Norm_5_128/batchnorm/sub×
 Batch_Norm_5_128/batchnorm/add_1AddV2$Batch_Norm_5_128/batchnorm/mul_1:z:0"Batch_Norm_5_128/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2"
 Batch_Norm_5_128/batchnorm/add_1³
/Batch_Norm_3_128/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       21
/Batch_Norm_3_128/moments/mean/reduction_indicesà
Batch_Norm_3_128/moments/meanMeanConv1D_3_128/IdentityN:output:08Batch_Norm_3_128/moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:*
	keep_dims(2
Batch_Norm_3_128/moments/mean´
%Batch_Norm_3_128/moments/StopGradientStopGradient&Batch_Norm_3_128/moments/mean:output:0*
T0*#
_output_shapes
:2'
%Batch_Norm_3_128/moments/StopGradientþ
*Batch_Norm_3_128/moments/SquaredDifferenceSquaredDifferenceConv1D_3_128/IdentityN:output:0.Batch_Norm_3_128/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2,
*Batch_Norm_3_128/moments/SquaredDifference»
3Batch_Norm_3_128/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       25
3Batch_Norm_3_128/moments/variance/reduction_indicesû
!Batch_Norm_3_128/moments/varianceMean.Batch_Norm_3_128/moments/SquaredDifference:z:0<Batch_Norm_3_128/moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:*
	keep_dims(2#
!Batch_Norm_3_128/moments/varianceµ
 Batch_Norm_3_128/moments/SqueezeSqueeze&Batch_Norm_3_128/moments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2"
 Batch_Norm_3_128/moments/Squeeze½
"Batch_Norm_3_128/moments/Squeeze_1Squeeze*Batch_Norm_3_128/moments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2$
"Batch_Norm_3_128/moments/Squeeze_1
&Batch_Norm_3_128/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2(
&Batch_Norm_3_128/AssignMovingAvg/decayØ
/Batch_Norm_3_128/AssignMovingAvg/ReadVariableOpReadVariableOp8batch_norm_3_128_assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype021
/Batch_Norm_3_128/AssignMovingAvg/ReadVariableOpÝ
$Batch_Norm_3_128/AssignMovingAvg/subSub7Batch_Norm_3_128/AssignMovingAvg/ReadVariableOp:value:0)Batch_Norm_3_128/moments/Squeeze:output:0*
T0*
_output_shapes	
:2&
$Batch_Norm_3_128/AssignMovingAvg/subÔ
$Batch_Norm_3_128/AssignMovingAvg/mulMul(Batch_Norm_3_128/AssignMovingAvg/sub:z:0/Batch_Norm_3_128/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:2&
$Batch_Norm_3_128/AssignMovingAvg/mul
 Batch_Norm_3_128/AssignMovingAvgAssignSubVariableOp8batch_norm_3_128_assignmovingavg_readvariableop_resource(Batch_Norm_3_128/AssignMovingAvg/mul:z:00^Batch_Norm_3_128/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02"
 Batch_Norm_3_128/AssignMovingAvg
(Batch_Norm_3_128/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2*
(Batch_Norm_3_128/AssignMovingAvg_1/decayÞ
1Batch_Norm_3_128/AssignMovingAvg_1/ReadVariableOpReadVariableOp:batch_norm_3_128_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype023
1Batch_Norm_3_128/AssignMovingAvg_1/ReadVariableOpå
&Batch_Norm_3_128/AssignMovingAvg_1/subSub9Batch_Norm_3_128/AssignMovingAvg_1/ReadVariableOp:value:0+Batch_Norm_3_128/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:2(
&Batch_Norm_3_128/AssignMovingAvg_1/subÜ
&Batch_Norm_3_128/AssignMovingAvg_1/mulMul*Batch_Norm_3_128/AssignMovingAvg_1/sub:z:01Batch_Norm_3_128/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:2(
&Batch_Norm_3_128/AssignMovingAvg_1/mul
"Batch_Norm_3_128/AssignMovingAvg_1AssignSubVariableOp:batch_norm_3_128_assignmovingavg_1_readvariableop_resource*Batch_Norm_3_128/AssignMovingAvg_1/mul:z:02^Batch_Norm_3_128/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02$
"Batch_Norm_3_128/AssignMovingAvg_1
 Batch_Norm_3_128/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2"
 Batch_Norm_3_128/batchnorm/add/yÇ
Batch_Norm_3_128/batchnorm/addAddV2+Batch_Norm_3_128/moments/Squeeze_1:output:0)Batch_Norm_3_128/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2 
Batch_Norm_3_128/batchnorm/add
 Batch_Norm_3_128/batchnorm/RsqrtRsqrt"Batch_Norm_3_128/batchnorm/add:z:0*
T0*
_output_shapes	
:2"
 Batch_Norm_3_128/batchnorm/RsqrtÒ
-Batch_Norm_3_128/batchnorm/mul/ReadVariableOpReadVariableOp6batch_norm_3_128_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02/
-Batch_Norm_3_128/batchnorm/mul/ReadVariableOpÊ
Batch_Norm_3_128/batchnorm/mulMul$Batch_Norm_3_128/batchnorm/Rsqrt:y:05Batch_Norm_3_128/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2 
Batch_Norm_3_128/batchnorm/mulÐ
 Batch_Norm_3_128/batchnorm/mul_1MulConv1D_3_128/IdentityN:output:0"Batch_Norm_3_128/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2"
 Batch_Norm_3_128/batchnorm/mul_1À
 Batch_Norm_3_128/batchnorm/mul_2Mul)Batch_Norm_3_128/moments/Squeeze:output:0"Batch_Norm_3_128/batchnorm/mul:z:0*
T0*
_output_shapes	
:2"
 Batch_Norm_3_128/batchnorm/mul_2Æ
)Batch_Norm_3_128/batchnorm/ReadVariableOpReadVariableOp2batch_norm_3_128_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02+
)Batch_Norm_3_128/batchnorm/ReadVariableOpÆ
Batch_Norm_3_128/batchnorm/subSub1Batch_Norm_3_128/batchnorm/ReadVariableOp:value:0$Batch_Norm_3_128/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2 
Batch_Norm_3_128/batchnorm/sub×
 Batch_Norm_3_128/batchnorm/add_1AddV2$Batch_Norm_3_128/batchnorm/mul_1:z:0"Batch_Norm_3_128/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2"
 Batch_Norm_3_128/batchnorm/add_1
"Conv1D_7_256/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2$
"Conv1D_7_256/conv1d/ExpandDims/dimå
Conv1D_7_256/conv1d/ExpandDims
ExpandDims$Batch_Norm_7_128/batchnorm/add_1:z:0+Conv1D_7_256/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2 
Conv1D_7_256/conv1d/ExpandDimsá
/Conv1D_7_256/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp8conv1d_7_256_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype021
/Conv1D_7_256/conv1d/ExpandDims_1/ReadVariableOp
$Conv1D_7_256/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Conv1D_7_256/conv1d/ExpandDims_1/dimí
 Conv1D_7_256/conv1d/ExpandDims_1
ExpandDims7Conv1D_7_256/conv1d/ExpandDims_1/ReadVariableOp:value:0-Conv1D_7_256/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2"
 Conv1D_7_256/conv1d/ExpandDims_1ô
Conv1D_7_256/conv1dConv2D'Conv1D_7_256/conv1d/ExpandDims:output:0)Conv1D_7_256/conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv1D_7_256/conv1dÃ
Conv1D_7_256/conv1d/SqueezeSqueezeConv1D_7_256/conv1d:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
Conv1D_7_256/conv1d/Squeeze´
#Conv1D_7_256/BiasAdd/ReadVariableOpReadVariableOp,conv1d_7_256_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#Conv1D_7_256/BiasAdd/ReadVariableOpÊ
Conv1D_7_256/BiasAddBiasAdd$Conv1D_7_256/conv1d/Squeeze:output:0+Conv1D_7_256/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_7_256/BiasAdd
Conv1D_7_256/SigmoidSigmoidConv1D_7_256/BiasAdd:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_7_256/Sigmoid¤
Conv1D_7_256/mulMulConv1D_7_256/BiasAdd:output:0Conv1D_7_256/Sigmoid:y:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_7_256/mul
Conv1D_7_256/IdentityIdentityConv1D_7_256/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_7_256/Identity
Conv1D_7_256/IdentityN	IdentityNConv1D_7_256/mul:z:0Conv1D_7_256/BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1916624173*V
_output_shapesD
B:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_7_256/IdentityN
"Conv1D_5_256/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2$
"Conv1D_5_256/conv1d/ExpandDims/dimå
Conv1D_5_256/conv1d/ExpandDims
ExpandDims$Batch_Norm_5_128/batchnorm/add_1:z:0+Conv1D_5_256/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2 
Conv1D_5_256/conv1d/ExpandDimsá
/Conv1D_5_256/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp8conv1d_5_256_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype021
/Conv1D_5_256/conv1d/ExpandDims_1/ReadVariableOp
$Conv1D_5_256/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Conv1D_5_256/conv1d/ExpandDims_1/dimí
 Conv1D_5_256/conv1d/ExpandDims_1
ExpandDims7Conv1D_5_256/conv1d/ExpandDims_1/ReadVariableOp:value:0-Conv1D_5_256/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2"
 Conv1D_5_256/conv1d/ExpandDims_1ô
Conv1D_5_256/conv1dConv2D'Conv1D_5_256/conv1d/ExpandDims:output:0)Conv1D_5_256/conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv1D_5_256/conv1dÃ
Conv1D_5_256/conv1d/SqueezeSqueezeConv1D_5_256/conv1d:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
Conv1D_5_256/conv1d/Squeeze´
#Conv1D_5_256/BiasAdd/ReadVariableOpReadVariableOp,conv1d_5_256_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#Conv1D_5_256/BiasAdd/ReadVariableOpÊ
Conv1D_5_256/BiasAddBiasAdd$Conv1D_5_256/conv1d/Squeeze:output:0+Conv1D_5_256/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_5_256/BiasAdd
Conv1D_5_256/SigmoidSigmoidConv1D_5_256/BiasAdd:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_5_256/Sigmoid¤
Conv1D_5_256/mulMulConv1D_5_256/BiasAdd:output:0Conv1D_5_256/Sigmoid:y:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_5_256/mul
Conv1D_5_256/IdentityIdentityConv1D_5_256/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_5_256/Identity
Conv1D_5_256/IdentityN	IdentityNConv1D_5_256/mul:z:0Conv1D_5_256/BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1916624190*V
_output_shapesD
B:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_5_256/IdentityN
"Conv1D_3_256/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2$
"Conv1D_3_256/conv1d/ExpandDims/dimå
Conv1D_3_256/conv1d/ExpandDims
ExpandDims$Batch_Norm_3_128/batchnorm/add_1:z:0+Conv1D_3_256/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2 
Conv1D_3_256/conv1d/ExpandDimsá
/Conv1D_3_256/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp8conv1d_3_256_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype021
/Conv1D_3_256/conv1d/ExpandDims_1/ReadVariableOp
$Conv1D_3_256/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Conv1D_3_256/conv1d/ExpandDims_1/dimí
 Conv1D_3_256/conv1d/ExpandDims_1
ExpandDims7Conv1D_3_256/conv1d/ExpandDims_1/ReadVariableOp:value:0-Conv1D_3_256/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2"
 Conv1D_3_256/conv1d/ExpandDims_1ô
Conv1D_3_256/conv1dConv2D'Conv1D_3_256/conv1d/ExpandDims:output:0)Conv1D_3_256/conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv1D_3_256/conv1dÃ
Conv1D_3_256/conv1d/SqueezeSqueezeConv1D_3_256/conv1d:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
Conv1D_3_256/conv1d/Squeeze´
#Conv1D_3_256/BiasAdd/ReadVariableOpReadVariableOp,conv1d_3_256_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#Conv1D_3_256/BiasAdd/ReadVariableOpÊ
Conv1D_3_256/BiasAddBiasAdd$Conv1D_3_256/conv1d/Squeeze:output:0+Conv1D_3_256/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_3_256/BiasAdd
Conv1D_3_256/SigmoidSigmoidConv1D_3_256/BiasAdd:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_3_256/Sigmoid¤
Conv1D_3_256/mulMulConv1D_3_256/BiasAdd:output:0Conv1D_3_256/Sigmoid:y:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_3_256/mul
Conv1D_3_256/IdentityIdentityConv1D_3_256/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_3_256/Identity
Conv1D_3_256/IdentityN	IdentityNConv1D_3_256/mul:z:0Conv1D_3_256/BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1916624207*V
_output_shapesD
B:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Conv1D_3_256/IdentityN³
/Batch_Norm_7_256/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       21
/Batch_Norm_7_256/moments/mean/reduction_indicesà
Batch_Norm_7_256/moments/meanMeanConv1D_7_256/IdentityN:output:08Batch_Norm_7_256/moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:*
	keep_dims(2
Batch_Norm_7_256/moments/mean´
%Batch_Norm_7_256/moments/StopGradientStopGradient&Batch_Norm_7_256/moments/mean:output:0*
T0*#
_output_shapes
:2'
%Batch_Norm_7_256/moments/StopGradientþ
*Batch_Norm_7_256/moments/SquaredDifferenceSquaredDifferenceConv1D_7_256/IdentityN:output:0.Batch_Norm_7_256/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2,
*Batch_Norm_7_256/moments/SquaredDifference»
3Batch_Norm_7_256/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       25
3Batch_Norm_7_256/moments/variance/reduction_indicesû
!Batch_Norm_7_256/moments/varianceMean.Batch_Norm_7_256/moments/SquaredDifference:z:0<Batch_Norm_7_256/moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:*
	keep_dims(2#
!Batch_Norm_7_256/moments/varianceµ
 Batch_Norm_7_256/moments/SqueezeSqueeze&Batch_Norm_7_256/moments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2"
 Batch_Norm_7_256/moments/Squeeze½
"Batch_Norm_7_256/moments/Squeeze_1Squeeze*Batch_Norm_7_256/moments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2$
"Batch_Norm_7_256/moments/Squeeze_1
&Batch_Norm_7_256/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2(
&Batch_Norm_7_256/AssignMovingAvg/decayØ
/Batch_Norm_7_256/AssignMovingAvg/ReadVariableOpReadVariableOp8batch_norm_7_256_assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype021
/Batch_Norm_7_256/AssignMovingAvg/ReadVariableOpÝ
$Batch_Norm_7_256/AssignMovingAvg/subSub7Batch_Norm_7_256/AssignMovingAvg/ReadVariableOp:value:0)Batch_Norm_7_256/moments/Squeeze:output:0*
T0*
_output_shapes	
:2&
$Batch_Norm_7_256/AssignMovingAvg/subÔ
$Batch_Norm_7_256/AssignMovingAvg/mulMul(Batch_Norm_7_256/AssignMovingAvg/sub:z:0/Batch_Norm_7_256/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:2&
$Batch_Norm_7_256/AssignMovingAvg/mul
 Batch_Norm_7_256/AssignMovingAvgAssignSubVariableOp8batch_norm_7_256_assignmovingavg_readvariableop_resource(Batch_Norm_7_256/AssignMovingAvg/mul:z:00^Batch_Norm_7_256/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02"
 Batch_Norm_7_256/AssignMovingAvg
(Batch_Norm_7_256/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2*
(Batch_Norm_7_256/AssignMovingAvg_1/decayÞ
1Batch_Norm_7_256/AssignMovingAvg_1/ReadVariableOpReadVariableOp:batch_norm_7_256_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype023
1Batch_Norm_7_256/AssignMovingAvg_1/ReadVariableOpå
&Batch_Norm_7_256/AssignMovingAvg_1/subSub9Batch_Norm_7_256/AssignMovingAvg_1/ReadVariableOp:value:0+Batch_Norm_7_256/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:2(
&Batch_Norm_7_256/AssignMovingAvg_1/subÜ
&Batch_Norm_7_256/AssignMovingAvg_1/mulMul*Batch_Norm_7_256/AssignMovingAvg_1/sub:z:01Batch_Norm_7_256/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:2(
&Batch_Norm_7_256/AssignMovingAvg_1/mul
"Batch_Norm_7_256/AssignMovingAvg_1AssignSubVariableOp:batch_norm_7_256_assignmovingavg_1_readvariableop_resource*Batch_Norm_7_256/AssignMovingAvg_1/mul:z:02^Batch_Norm_7_256/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02$
"Batch_Norm_7_256/AssignMovingAvg_1
 Batch_Norm_7_256/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2"
 Batch_Norm_7_256/batchnorm/add/yÇ
Batch_Norm_7_256/batchnorm/addAddV2+Batch_Norm_7_256/moments/Squeeze_1:output:0)Batch_Norm_7_256/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2 
Batch_Norm_7_256/batchnorm/add
 Batch_Norm_7_256/batchnorm/RsqrtRsqrt"Batch_Norm_7_256/batchnorm/add:z:0*
T0*
_output_shapes	
:2"
 Batch_Norm_7_256/batchnorm/RsqrtÒ
-Batch_Norm_7_256/batchnorm/mul/ReadVariableOpReadVariableOp6batch_norm_7_256_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02/
-Batch_Norm_7_256/batchnorm/mul/ReadVariableOpÊ
Batch_Norm_7_256/batchnorm/mulMul$Batch_Norm_7_256/batchnorm/Rsqrt:y:05Batch_Norm_7_256/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2 
Batch_Norm_7_256/batchnorm/mulÐ
 Batch_Norm_7_256/batchnorm/mul_1MulConv1D_7_256/IdentityN:output:0"Batch_Norm_7_256/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2"
 Batch_Norm_7_256/batchnorm/mul_1À
 Batch_Norm_7_256/batchnorm/mul_2Mul)Batch_Norm_7_256/moments/Squeeze:output:0"Batch_Norm_7_256/batchnorm/mul:z:0*
T0*
_output_shapes	
:2"
 Batch_Norm_7_256/batchnorm/mul_2Æ
)Batch_Norm_7_256/batchnorm/ReadVariableOpReadVariableOp2batch_norm_7_256_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02+
)Batch_Norm_7_256/batchnorm/ReadVariableOpÆ
Batch_Norm_7_256/batchnorm/subSub1Batch_Norm_7_256/batchnorm/ReadVariableOp:value:0$Batch_Norm_7_256/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2 
Batch_Norm_7_256/batchnorm/sub×
 Batch_Norm_7_256/batchnorm/add_1AddV2$Batch_Norm_7_256/batchnorm/mul_1:z:0"Batch_Norm_7_256/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2"
 Batch_Norm_7_256/batchnorm/add_1³
/Batch_Norm_5_256/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       21
/Batch_Norm_5_256/moments/mean/reduction_indicesà
Batch_Norm_5_256/moments/meanMeanConv1D_5_256/IdentityN:output:08Batch_Norm_5_256/moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:*
	keep_dims(2
Batch_Norm_5_256/moments/mean´
%Batch_Norm_5_256/moments/StopGradientStopGradient&Batch_Norm_5_256/moments/mean:output:0*
T0*#
_output_shapes
:2'
%Batch_Norm_5_256/moments/StopGradientþ
*Batch_Norm_5_256/moments/SquaredDifferenceSquaredDifferenceConv1D_5_256/IdentityN:output:0.Batch_Norm_5_256/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2,
*Batch_Norm_5_256/moments/SquaredDifference»
3Batch_Norm_5_256/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       25
3Batch_Norm_5_256/moments/variance/reduction_indicesû
!Batch_Norm_5_256/moments/varianceMean.Batch_Norm_5_256/moments/SquaredDifference:z:0<Batch_Norm_5_256/moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:*
	keep_dims(2#
!Batch_Norm_5_256/moments/varianceµ
 Batch_Norm_5_256/moments/SqueezeSqueeze&Batch_Norm_5_256/moments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2"
 Batch_Norm_5_256/moments/Squeeze½
"Batch_Norm_5_256/moments/Squeeze_1Squeeze*Batch_Norm_5_256/moments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2$
"Batch_Norm_5_256/moments/Squeeze_1
&Batch_Norm_5_256/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2(
&Batch_Norm_5_256/AssignMovingAvg/decayØ
/Batch_Norm_5_256/AssignMovingAvg/ReadVariableOpReadVariableOp8batch_norm_5_256_assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype021
/Batch_Norm_5_256/AssignMovingAvg/ReadVariableOpÝ
$Batch_Norm_5_256/AssignMovingAvg/subSub7Batch_Norm_5_256/AssignMovingAvg/ReadVariableOp:value:0)Batch_Norm_5_256/moments/Squeeze:output:0*
T0*
_output_shapes	
:2&
$Batch_Norm_5_256/AssignMovingAvg/subÔ
$Batch_Norm_5_256/AssignMovingAvg/mulMul(Batch_Norm_5_256/AssignMovingAvg/sub:z:0/Batch_Norm_5_256/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:2&
$Batch_Norm_5_256/AssignMovingAvg/mul
 Batch_Norm_5_256/AssignMovingAvgAssignSubVariableOp8batch_norm_5_256_assignmovingavg_readvariableop_resource(Batch_Norm_5_256/AssignMovingAvg/mul:z:00^Batch_Norm_5_256/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02"
 Batch_Norm_5_256/AssignMovingAvg
(Batch_Norm_5_256/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2*
(Batch_Norm_5_256/AssignMovingAvg_1/decayÞ
1Batch_Norm_5_256/AssignMovingAvg_1/ReadVariableOpReadVariableOp:batch_norm_5_256_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype023
1Batch_Norm_5_256/AssignMovingAvg_1/ReadVariableOpå
&Batch_Norm_5_256/AssignMovingAvg_1/subSub9Batch_Norm_5_256/AssignMovingAvg_1/ReadVariableOp:value:0+Batch_Norm_5_256/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:2(
&Batch_Norm_5_256/AssignMovingAvg_1/subÜ
&Batch_Norm_5_256/AssignMovingAvg_1/mulMul*Batch_Norm_5_256/AssignMovingAvg_1/sub:z:01Batch_Norm_5_256/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:2(
&Batch_Norm_5_256/AssignMovingAvg_1/mul
"Batch_Norm_5_256/AssignMovingAvg_1AssignSubVariableOp:batch_norm_5_256_assignmovingavg_1_readvariableop_resource*Batch_Norm_5_256/AssignMovingAvg_1/mul:z:02^Batch_Norm_5_256/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02$
"Batch_Norm_5_256/AssignMovingAvg_1
 Batch_Norm_5_256/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2"
 Batch_Norm_5_256/batchnorm/add/yÇ
Batch_Norm_5_256/batchnorm/addAddV2+Batch_Norm_5_256/moments/Squeeze_1:output:0)Batch_Norm_5_256/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2 
Batch_Norm_5_256/batchnorm/add
 Batch_Norm_5_256/batchnorm/RsqrtRsqrt"Batch_Norm_5_256/batchnorm/add:z:0*
T0*
_output_shapes	
:2"
 Batch_Norm_5_256/batchnorm/RsqrtÒ
-Batch_Norm_5_256/batchnorm/mul/ReadVariableOpReadVariableOp6batch_norm_5_256_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02/
-Batch_Norm_5_256/batchnorm/mul/ReadVariableOpÊ
Batch_Norm_5_256/batchnorm/mulMul$Batch_Norm_5_256/batchnorm/Rsqrt:y:05Batch_Norm_5_256/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2 
Batch_Norm_5_256/batchnorm/mulÐ
 Batch_Norm_5_256/batchnorm/mul_1MulConv1D_5_256/IdentityN:output:0"Batch_Norm_5_256/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2"
 Batch_Norm_5_256/batchnorm/mul_1À
 Batch_Norm_5_256/batchnorm/mul_2Mul)Batch_Norm_5_256/moments/Squeeze:output:0"Batch_Norm_5_256/batchnorm/mul:z:0*
T0*
_output_shapes	
:2"
 Batch_Norm_5_256/batchnorm/mul_2Æ
)Batch_Norm_5_256/batchnorm/ReadVariableOpReadVariableOp2batch_norm_5_256_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02+
)Batch_Norm_5_256/batchnorm/ReadVariableOpÆ
Batch_Norm_5_256/batchnorm/subSub1Batch_Norm_5_256/batchnorm/ReadVariableOp:value:0$Batch_Norm_5_256/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2 
Batch_Norm_5_256/batchnorm/sub×
 Batch_Norm_5_256/batchnorm/add_1AddV2$Batch_Norm_5_256/batchnorm/mul_1:z:0"Batch_Norm_5_256/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2"
 Batch_Norm_5_256/batchnorm/add_1³
/Batch_Norm_3_256/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       21
/Batch_Norm_3_256/moments/mean/reduction_indicesà
Batch_Norm_3_256/moments/meanMeanConv1D_3_256/IdentityN:output:08Batch_Norm_3_256/moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:*
	keep_dims(2
Batch_Norm_3_256/moments/mean´
%Batch_Norm_3_256/moments/StopGradientStopGradient&Batch_Norm_3_256/moments/mean:output:0*
T0*#
_output_shapes
:2'
%Batch_Norm_3_256/moments/StopGradientþ
*Batch_Norm_3_256/moments/SquaredDifferenceSquaredDifferenceConv1D_3_256/IdentityN:output:0.Batch_Norm_3_256/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2,
*Batch_Norm_3_256/moments/SquaredDifference»
3Batch_Norm_3_256/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       25
3Batch_Norm_3_256/moments/variance/reduction_indicesû
!Batch_Norm_3_256/moments/varianceMean.Batch_Norm_3_256/moments/SquaredDifference:z:0<Batch_Norm_3_256/moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:*
	keep_dims(2#
!Batch_Norm_3_256/moments/varianceµ
 Batch_Norm_3_256/moments/SqueezeSqueeze&Batch_Norm_3_256/moments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2"
 Batch_Norm_3_256/moments/Squeeze½
"Batch_Norm_3_256/moments/Squeeze_1Squeeze*Batch_Norm_3_256/moments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2$
"Batch_Norm_3_256/moments/Squeeze_1
&Batch_Norm_3_256/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2(
&Batch_Norm_3_256/AssignMovingAvg/decayØ
/Batch_Norm_3_256/AssignMovingAvg/ReadVariableOpReadVariableOp8batch_norm_3_256_assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype021
/Batch_Norm_3_256/AssignMovingAvg/ReadVariableOpÝ
$Batch_Norm_3_256/AssignMovingAvg/subSub7Batch_Norm_3_256/AssignMovingAvg/ReadVariableOp:value:0)Batch_Norm_3_256/moments/Squeeze:output:0*
T0*
_output_shapes	
:2&
$Batch_Norm_3_256/AssignMovingAvg/subÔ
$Batch_Norm_3_256/AssignMovingAvg/mulMul(Batch_Norm_3_256/AssignMovingAvg/sub:z:0/Batch_Norm_3_256/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:2&
$Batch_Norm_3_256/AssignMovingAvg/mul
 Batch_Norm_3_256/AssignMovingAvgAssignSubVariableOp8batch_norm_3_256_assignmovingavg_readvariableop_resource(Batch_Norm_3_256/AssignMovingAvg/mul:z:00^Batch_Norm_3_256/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02"
 Batch_Norm_3_256/AssignMovingAvg
(Batch_Norm_3_256/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2*
(Batch_Norm_3_256/AssignMovingAvg_1/decayÞ
1Batch_Norm_3_256/AssignMovingAvg_1/ReadVariableOpReadVariableOp:batch_norm_3_256_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype023
1Batch_Norm_3_256/AssignMovingAvg_1/ReadVariableOpå
&Batch_Norm_3_256/AssignMovingAvg_1/subSub9Batch_Norm_3_256/AssignMovingAvg_1/ReadVariableOp:value:0+Batch_Norm_3_256/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:2(
&Batch_Norm_3_256/AssignMovingAvg_1/subÜ
&Batch_Norm_3_256/AssignMovingAvg_1/mulMul*Batch_Norm_3_256/AssignMovingAvg_1/sub:z:01Batch_Norm_3_256/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:2(
&Batch_Norm_3_256/AssignMovingAvg_1/mul
"Batch_Norm_3_256/AssignMovingAvg_1AssignSubVariableOp:batch_norm_3_256_assignmovingavg_1_readvariableop_resource*Batch_Norm_3_256/AssignMovingAvg_1/mul:z:02^Batch_Norm_3_256/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02$
"Batch_Norm_3_256/AssignMovingAvg_1
 Batch_Norm_3_256/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2"
 Batch_Norm_3_256/batchnorm/add/yÇ
Batch_Norm_3_256/batchnorm/addAddV2+Batch_Norm_3_256/moments/Squeeze_1:output:0)Batch_Norm_3_256/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2 
Batch_Norm_3_256/batchnorm/add
 Batch_Norm_3_256/batchnorm/RsqrtRsqrt"Batch_Norm_3_256/batchnorm/add:z:0*
T0*
_output_shapes	
:2"
 Batch_Norm_3_256/batchnorm/RsqrtÒ
-Batch_Norm_3_256/batchnorm/mul/ReadVariableOpReadVariableOp6batch_norm_3_256_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02/
-Batch_Norm_3_256/batchnorm/mul/ReadVariableOpÊ
Batch_Norm_3_256/batchnorm/mulMul$Batch_Norm_3_256/batchnorm/Rsqrt:y:05Batch_Norm_3_256/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2 
Batch_Norm_3_256/batchnorm/mulÐ
 Batch_Norm_3_256/batchnorm/mul_1MulConv1D_3_256/IdentityN:output:0"Batch_Norm_3_256/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2"
 Batch_Norm_3_256/batchnorm/mul_1À
 Batch_Norm_3_256/batchnorm/mul_2Mul)Batch_Norm_3_256/moments/Squeeze:output:0"Batch_Norm_3_256/batchnorm/mul:z:0*
T0*
_output_shapes	
:2"
 Batch_Norm_3_256/batchnorm/mul_2Æ
)Batch_Norm_3_256/batchnorm/ReadVariableOpReadVariableOp2batch_norm_3_256_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02+
)Batch_Norm_3_256/batchnorm/ReadVariableOpÆ
Batch_Norm_3_256/batchnorm/subSub1Batch_Norm_3_256/batchnorm/ReadVariableOp:value:0$Batch_Norm_3_256/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2 
Batch_Norm_3_256/batchnorm/sub×
 Batch_Norm_3_256/batchnorm/add_1AddV2$Batch_Norm_3_256/batchnorm/mul_1:z:0"Batch_Norm_3_256/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2"
 Batch_Norm_3_256/batchnorm/add_1
 Max_Pool_3/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2"
 Max_Pool_3/Max/reduction_indices«
Max_Pool_3/MaxMax$Batch_Norm_3_256/batchnorm/add_1:z:0)Max_Pool_3/Max/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Max_Pool_3/Max
 Max_Pool_5/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2"
 Max_Pool_5/Max/reduction_indices«
Max_Pool_5/MaxMax$Batch_Norm_5_256/batchnorm/add_1:z:0)Max_Pool_5/Max/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Max_Pool_5/Max
 Max_Pool_7/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2"
 Max_Pool_7/Max/reduction_indices«
Max_Pool_7/MaxMax$Batch_Norm_7_256/batchnorm/add_1:z:0)Max_Pool_7/Max/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Max_Pool_7/Max~
Pool_Concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
Pool_Concatenate/concat/axisì
Pool_Concatenate/concatConcatV2Max_Pool_3/Max:output:0Max_Pool_5/Max:output:0Max_Pool_7/Max:output:0%Pool_Concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Pool_Concatenate/concat
!Concatenate_Dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!Concatenate_Dropout/dropout/ConstÊ
Concatenate_Dropout/dropout/MulMul Pool_Concatenate/concat:output:0*Concatenate_Dropout/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
Concatenate_Dropout/dropout/Mul
!Concatenate_Dropout/dropout/ShapeShape Pool_Concatenate/concat:output:0*
T0*
_output_shapes
:2#
!Concatenate_Dropout/dropout/Shapeñ
8Concatenate_Dropout/dropout/random_uniform/RandomUniformRandomUniform*Concatenate_Dropout/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02:
8Concatenate_Dropout/dropout/random_uniform/RandomUniform
*Concatenate_Dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2,
*Concatenate_Dropout/dropout/GreaterEqual/y
(Concatenate_Dropout/dropout/GreaterEqualGreaterEqualAConcatenate_Dropout/dropout/random_uniform/RandomUniform:output:03Concatenate_Dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(Concatenate_Dropout/dropout/GreaterEqual¼
 Concatenate_Dropout/dropout/CastCast,Concatenate_Dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 Concatenate_Dropout/dropout/CastË
!Concatenate_Dropout/dropout/Mul_1Mul#Concatenate_Dropout/dropout/Mul:z:0$Concatenate_Dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!Concatenate_Dropout/dropout/Mul_1¯
 Prediction/MatMul/ReadVariableOpReadVariableOp)prediction_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 Prediction/MatMul/ReadVariableOp³
Prediction/MatMulMatMul%Concatenate_Dropout/dropout/Mul_1:z:0(Prediction/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Prediction/MatMul­
!Prediction/BiasAdd/ReadVariableOpReadVariableOp*prediction_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!Prediction/BiasAdd/ReadVariableOp­
Prediction/BiasAddBiasAddPrediction/MatMul:product:0)Prediction/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Prediction/BiasAdd
Prediction/SigmoidSigmoidPrediction/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Prediction/Sigmoidý
IdentityIdentityPrediction/Sigmoid:y:0!^Batch_Norm_3_128/AssignMovingAvg0^Batch_Norm_3_128/AssignMovingAvg/ReadVariableOp#^Batch_Norm_3_128/AssignMovingAvg_12^Batch_Norm_3_128/AssignMovingAvg_1/ReadVariableOp*^Batch_Norm_3_128/batchnorm/ReadVariableOp.^Batch_Norm_3_128/batchnorm/mul/ReadVariableOp!^Batch_Norm_3_256/AssignMovingAvg0^Batch_Norm_3_256/AssignMovingAvg/ReadVariableOp#^Batch_Norm_3_256/AssignMovingAvg_12^Batch_Norm_3_256/AssignMovingAvg_1/ReadVariableOp*^Batch_Norm_3_256/batchnorm/ReadVariableOp.^Batch_Norm_3_256/batchnorm/mul/ReadVariableOp!^Batch_Norm_5_128/AssignMovingAvg0^Batch_Norm_5_128/AssignMovingAvg/ReadVariableOp#^Batch_Norm_5_128/AssignMovingAvg_12^Batch_Norm_5_128/AssignMovingAvg_1/ReadVariableOp*^Batch_Norm_5_128/batchnorm/ReadVariableOp.^Batch_Norm_5_128/batchnorm/mul/ReadVariableOp!^Batch_Norm_5_256/AssignMovingAvg0^Batch_Norm_5_256/AssignMovingAvg/ReadVariableOp#^Batch_Norm_5_256/AssignMovingAvg_12^Batch_Norm_5_256/AssignMovingAvg_1/ReadVariableOp*^Batch_Norm_5_256/batchnorm/ReadVariableOp.^Batch_Norm_5_256/batchnorm/mul/ReadVariableOp!^Batch_Norm_7_128/AssignMovingAvg0^Batch_Norm_7_128/AssignMovingAvg/ReadVariableOp#^Batch_Norm_7_128/AssignMovingAvg_12^Batch_Norm_7_128/AssignMovingAvg_1/ReadVariableOp*^Batch_Norm_7_128/batchnorm/ReadVariableOp.^Batch_Norm_7_128/batchnorm/mul/ReadVariableOp!^Batch_Norm_7_256/AssignMovingAvg0^Batch_Norm_7_256/AssignMovingAvg/ReadVariableOp#^Batch_Norm_7_256/AssignMovingAvg_12^Batch_Norm_7_256/AssignMovingAvg_1/ReadVariableOp*^Batch_Norm_7_256/batchnorm/ReadVariableOp.^Batch_Norm_7_256/batchnorm/mul/ReadVariableOp$^Conv1D_3_128/BiasAdd/ReadVariableOp0^Conv1D_3_128/conv1d/ExpandDims_1/ReadVariableOp$^Conv1D_3_256/BiasAdd/ReadVariableOp0^Conv1D_3_256/conv1d/ExpandDims_1/ReadVariableOp$^Conv1D_5_128/BiasAdd/ReadVariableOp0^Conv1D_5_128/conv1d/ExpandDims_1/ReadVariableOp$^Conv1D_5_256/BiasAdd/ReadVariableOp0^Conv1D_5_256/conv1d/ExpandDims_1/ReadVariableOp$^Conv1D_7_128/BiasAdd/ReadVariableOp0^Conv1D_7_128/conv1d/ExpandDims_1/ReadVariableOp$^Conv1D_7_256/BiasAdd/ReadVariableOp0^Conv1D_7_256/conv1d/ExpandDims_1/ReadVariableOp"^Prediction/BiasAdd/ReadVariableOp!^Prediction/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapeso
m:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 Batch_Norm_3_128/AssignMovingAvg Batch_Norm_3_128/AssignMovingAvg2b
/Batch_Norm_3_128/AssignMovingAvg/ReadVariableOp/Batch_Norm_3_128/AssignMovingAvg/ReadVariableOp2H
"Batch_Norm_3_128/AssignMovingAvg_1"Batch_Norm_3_128/AssignMovingAvg_12f
1Batch_Norm_3_128/AssignMovingAvg_1/ReadVariableOp1Batch_Norm_3_128/AssignMovingAvg_1/ReadVariableOp2V
)Batch_Norm_3_128/batchnorm/ReadVariableOp)Batch_Norm_3_128/batchnorm/ReadVariableOp2^
-Batch_Norm_3_128/batchnorm/mul/ReadVariableOp-Batch_Norm_3_128/batchnorm/mul/ReadVariableOp2D
 Batch_Norm_3_256/AssignMovingAvg Batch_Norm_3_256/AssignMovingAvg2b
/Batch_Norm_3_256/AssignMovingAvg/ReadVariableOp/Batch_Norm_3_256/AssignMovingAvg/ReadVariableOp2H
"Batch_Norm_3_256/AssignMovingAvg_1"Batch_Norm_3_256/AssignMovingAvg_12f
1Batch_Norm_3_256/AssignMovingAvg_1/ReadVariableOp1Batch_Norm_3_256/AssignMovingAvg_1/ReadVariableOp2V
)Batch_Norm_3_256/batchnorm/ReadVariableOp)Batch_Norm_3_256/batchnorm/ReadVariableOp2^
-Batch_Norm_3_256/batchnorm/mul/ReadVariableOp-Batch_Norm_3_256/batchnorm/mul/ReadVariableOp2D
 Batch_Norm_5_128/AssignMovingAvg Batch_Norm_5_128/AssignMovingAvg2b
/Batch_Norm_5_128/AssignMovingAvg/ReadVariableOp/Batch_Norm_5_128/AssignMovingAvg/ReadVariableOp2H
"Batch_Norm_5_128/AssignMovingAvg_1"Batch_Norm_5_128/AssignMovingAvg_12f
1Batch_Norm_5_128/AssignMovingAvg_1/ReadVariableOp1Batch_Norm_5_128/AssignMovingAvg_1/ReadVariableOp2V
)Batch_Norm_5_128/batchnorm/ReadVariableOp)Batch_Norm_5_128/batchnorm/ReadVariableOp2^
-Batch_Norm_5_128/batchnorm/mul/ReadVariableOp-Batch_Norm_5_128/batchnorm/mul/ReadVariableOp2D
 Batch_Norm_5_256/AssignMovingAvg Batch_Norm_5_256/AssignMovingAvg2b
/Batch_Norm_5_256/AssignMovingAvg/ReadVariableOp/Batch_Norm_5_256/AssignMovingAvg/ReadVariableOp2H
"Batch_Norm_5_256/AssignMovingAvg_1"Batch_Norm_5_256/AssignMovingAvg_12f
1Batch_Norm_5_256/AssignMovingAvg_1/ReadVariableOp1Batch_Norm_5_256/AssignMovingAvg_1/ReadVariableOp2V
)Batch_Norm_5_256/batchnorm/ReadVariableOp)Batch_Norm_5_256/batchnorm/ReadVariableOp2^
-Batch_Norm_5_256/batchnorm/mul/ReadVariableOp-Batch_Norm_5_256/batchnorm/mul/ReadVariableOp2D
 Batch_Norm_7_128/AssignMovingAvg Batch_Norm_7_128/AssignMovingAvg2b
/Batch_Norm_7_128/AssignMovingAvg/ReadVariableOp/Batch_Norm_7_128/AssignMovingAvg/ReadVariableOp2H
"Batch_Norm_7_128/AssignMovingAvg_1"Batch_Norm_7_128/AssignMovingAvg_12f
1Batch_Norm_7_128/AssignMovingAvg_1/ReadVariableOp1Batch_Norm_7_128/AssignMovingAvg_1/ReadVariableOp2V
)Batch_Norm_7_128/batchnorm/ReadVariableOp)Batch_Norm_7_128/batchnorm/ReadVariableOp2^
-Batch_Norm_7_128/batchnorm/mul/ReadVariableOp-Batch_Norm_7_128/batchnorm/mul/ReadVariableOp2D
 Batch_Norm_7_256/AssignMovingAvg Batch_Norm_7_256/AssignMovingAvg2b
/Batch_Norm_7_256/AssignMovingAvg/ReadVariableOp/Batch_Norm_7_256/AssignMovingAvg/ReadVariableOp2H
"Batch_Norm_7_256/AssignMovingAvg_1"Batch_Norm_7_256/AssignMovingAvg_12f
1Batch_Norm_7_256/AssignMovingAvg_1/ReadVariableOp1Batch_Norm_7_256/AssignMovingAvg_1/ReadVariableOp2V
)Batch_Norm_7_256/batchnorm/ReadVariableOp)Batch_Norm_7_256/batchnorm/ReadVariableOp2^
-Batch_Norm_7_256/batchnorm/mul/ReadVariableOp-Batch_Norm_7_256/batchnorm/mul/ReadVariableOp2J
#Conv1D_3_128/BiasAdd/ReadVariableOp#Conv1D_3_128/BiasAdd/ReadVariableOp2b
/Conv1D_3_128/conv1d/ExpandDims_1/ReadVariableOp/Conv1D_3_128/conv1d/ExpandDims_1/ReadVariableOp2J
#Conv1D_3_256/BiasAdd/ReadVariableOp#Conv1D_3_256/BiasAdd/ReadVariableOp2b
/Conv1D_3_256/conv1d/ExpandDims_1/ReadVariableOp/Conv1D_3_256/conv1d/ExpandDims_1/ReadVariableOp2J
#Conv1D_5_128/BiasAdd/ReadVariableOp#Conv1D_5_128/BiasAdd/ReadVariableOp2b
/Conv1D_5_128/conv1d/ExpandDims_1/ReadVariableOp/Conv1D_5_128/conv1d/ExpandDims_1/ReadVariableOp2J
#Conv1D_5_256/BiasAdd/ReadVariableOp#Conv1D_5_256/BiasAdd/ReadVariableOp2b
/Conv1D_5_256/conv1d/ExpandDims_1/ReadVariableOp/Conv1D_5_256/conv1d/ExpandDims_1/ReadVariableOp2J
#Conv1D_7_128/BiasAdd/ReadVariableOp#Conv1D_7_128/BiasAdd/ReadVariableOp2b
/Conv1D_7_128/conv1d/ExpandDims_1/ReadVariableOp/Conv1D_7_128/conv1d/ExpandDims_1/ReadVariableOp2J
#Conv1D_7_256/BiasAdd/ReadVariableOp#Conv1D_7_256/BiasAdd/ReadVariableOp2b
/Conv1D_7_256/conv1d/ExpandDims_1/ReadVariableOp/Conv1D_7_256/conv1d/ExpandDims_1/ReadVariableOp2F
!Prediction/BiasAdd/ReadVariableOp!Prediction/BiasAdd/ReadVariableOp2D
 Prediction/MatMul/ReadVariableOp Prediction/MatMul/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ð
	
1__inference_Tree_Encoder_layer_call_fn_1916623348

inputs
unknown:
	unknown_0:	!
	unknown_1:
	unknown_2:	!
	unknown_3:
	unknown_4:	
	unknown_5:	
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:	

unknown_11:	

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	"

unknown_17:

unknown_18:	"

unknown_19:

unknown_20:	"

unknown_21:

unknown_22:	

unknown_23:	

unknown_24:	

unknown_25:	

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:	

unknown_30:	

unknown_31:	

unknown_32:	

unknown_33:	

unknown_34:	

unknown_35:	

unknown_36:
identity¢StatefulPartitionedCallâ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*<
_read_only_resource_inputs
	
 #$%&*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_Tree_Encoder_layer_call_and_return_conditional_losses_19166231882
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapeso
m:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameInputs
ü
 
L__inference_Conv1D_7_256_layer_call_and_return_conditional_losses_1916624774

inputsC
+conv1d_expanddims_1_readvariableop_resource:.
biasadd_readvariableop_resource:	

identity_1¢BiasAdd/ReadVariableOp¢"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims/dim 
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
conv1d/ExpandDimsº
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim¹
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv1d/ExpandDims_1À
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAddo
SigmoidSigmoidBiasAdd:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
Sigmoidp
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
muli
IdentityIdentitymul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

IdentityÕ
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1916624767*V
_output_shapesD
B:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	IdentityN¶

Identity_1IdentityIdentityN:output:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü*
í
P__inference_Batch_Norm_7_128_layer_call_and_return_conditional_losses_1916624684

inputs6
'assignmovingavg_readvariableop_resource:	8
)assignmovingavg_1_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	0
!batchnorm_readvariableop_resource:	
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:2
moments/StopGradient²
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices·
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay¥
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg/mul¿
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay«
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp¡
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ
³
P__inference_Batch_Norm_7_256_layer_call_and_return_conditional_losses_1916624980

inputs0
!batchnorm_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	2
#batchnorm_readvariableop_1_resource:	2
#batchnorm_readvariableop_2_resource:	
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1é
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ã
o
5__inference_Pool_Concatenate_layer_call_fn_1916625021
inputs_0
inputs_1
inputs_2
identityê
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_Pool_Concatenate_layer_call_and_return_conditional_losses_19166227762
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:R N
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/2
ü*
í
P__inference_Batch_Norm_5_128_layer_call_and_return_conditional_losses_1916621770

inputs6
'assignmovingavg_readvariableop_resource:	8
)assignmovingavg_1_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	0
!batchnorm_readvariableop_resource:	
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:2
moments/StopGradient²
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices·
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay¥
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg/mul¿
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay«
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp¡
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª
p
Q__inference_Embedding_Dropout_layer_call_and_return_conditional_losses_1916623003

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Const
dropout/MulMulinputsdropout/Const:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeÂ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yÌ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1s
IdentityIdentitydropout/Mul_1:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
k

L__inference_Tree_Encoder_layer_call_and_return_conditional_losses_1916623447

inputs/
conv1d_7_128_1916623352:&
conv1d_7_128_1916623354:	/
conv1d_5_128_1916623357:&
conv1d_5_128_1916623359:	/
conv1d_3_128_1916623362:&
conv1d_3_128_1916623364:	*
batch_norm_7_128_1916623367:	*
batch_norm_7_128_1916623369:	*
batch_norm_7_128_1916623371:	*
batch_norm_7_128_1916623373:	*
batch_norm_5_128_1916623376:	*
batch_norm_5_128_1916623378:	*
batch_norm_5_128_1916623380:	*
batch_norm_5_128_1916623382:	*
batch_norm_3_128_1916623385:	*
batch_norm_3_128_1916623387:	*
batch_norm_3_128_1916623389:	*
batch_norm_3_128_1916623391:	/
conv1d_7_256_1916623394:&
conv1d_7_256_1916623396:	/
conv1d_5_256_1916623399:&
conv1d_5_256_1916623401:	/
conv1d_3_256_1916623404:&
conv1d_3_256_1916623406:	*
batch_norm_7_256_1916623409:	*
batch_norm_7_256_1916623411:	*
batch_norm_7_256_1916623413:	*
batch_norm_7_256_1916623415:	*
batch_norm_5_256_1916623418:	*
batch_norm_5_256_1916623420:	*
batch_norm_5_256_1916623422:	*
batch_norm_5_256_1916623424:	*
batch_norm_3_256_1916623427:	*
batch_norm_3_256_1916623429:	*
batch_norm_3_256_1916623431:	*
batch_norm_3_256_1916623433:	(
prediction_1916623441:	#
prediction_1916623443:
identity¢(Batch_Norm_3_128/StatefulPartitionedCall¢(Batch_Norm_3_256/StatefulPartitionedCall¢(Batch_Norm_5_128/StatefulPartitionedCall¢(Batch_Norm_5_256/StatefulPartitionedCall¢(Batch_Norm_7_128/StatefulPartitionedCall¢(Batch_Norm_7_256/StatefulPartitionedCall¢$Conv1D_3_128/StatefulPartitionedCall¢$Conv1D_3_256/StatefulPartitionedCall¢$Conv1D_5_128/StatefulPartitionedCall¢$Conv1D_5_256/StatefulPartitionedCall¢$Conv1D_7_128/StatefulPartitionedCall¢$Conv1D_7_256/StatefulPartitionedCall¢"Prediction/StatefulPartitionedCall
!Embedding_Dropout/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_Embedding_Dropout_layer_call_and_return_conditional_losses_19166225472#
!Embedding_Dropout/PartitionedCallé
$Conv1D_7_128/StatefulPartitionedCallStatefulPartitionedCall*Embedding_Dropout/PartitionedCall:output:0conv1d_7_128_1916623352conv1d_7_128_1916623354*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_Conv1D_7_128_layer_call_and_return_conditional_losses_19166225702&
$Conv1D_7_128/StatefulPartitionedCallé
$Conv1D_5_128/StatefulPartitionedCallStatefulPartitionedCall*Embedding_Dropout/PartitionedCall:output:0conv1d_5_128_1916623357conv1d_5_128_1916623359*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_Conv1D_5_128_layer_call_and_return_conditional_losses_19166225972&
$Conv1D_5_128/StatefulPartitionedCallé
$Conv1D_3_128/StatefulPartitionedCallStatefulPartitionedCall*Embedding_Dropout/PartitionedCall:output:0conv1d_3_128_1916623362conv1d_3_128_1916623364*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_Conv1D_3_128_layer_call_and_return_conditional_losses_19166226242&
$Conv1D_3_128/StatefulPartitionedCall¾
(Batch_Norm_7_128/StatefulPartitionedCallStatefulPartitionedCall-Conv1D_7_128/StatefulPartitionedCall:output:0batch_norm_7_128_1916623367batch_norm_7_128_1916623369batch_norm_7_128_1916623371batch_norm_7_128_1916623373*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_Batch_Norm_7_128_layer_call_and_return_conditional_losses_19166218722*
(Batch_Norm_7_128/StatefulPartitionedCall¾
(Batch_Norm_5_128/StatefulPartitionedCallStatefulPartitionedCall-Conv1D_5_128/StatefulPartitionedCall:output:0batch_norm_5_128_1916623376batch_norm_5_128_1916623378batch_norm_5_128_1916623380batch_norm_5_128_1916623382*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_Batch_Norm_5_128_layer_call_and_return_conditional_losses_19166217102*
(Batch_Norm_5_128/StatefulPartitionedCall¾
(Batch_Norm_3_128/StatefulPartitionedCallStatefulPartitionedCall-Conv1D_3_128/StatefulPartitionedCall:output:0batch_norm_3_128_1916623385batch_norm_3_128_1916623387batch_norm_3_128_1916623389batch_norm_3_128_1916623391*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_Batch_Norm_3_128_layer_call_and_return_conditional_losses_19166215482*
(Batch_Norm_3_128/StatefulPartitionedCallð
$Conv1D_7_256/StatefulPartitionedCallStatefulPartitionedCall1Batch_Norm_7_128/StatefulPartitionedCall:output:0conv1d_7_256_1916623394conv1d_7_256_1916623396*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_Conv1D_7_256_layer_call_and_return_conditional_losses_19166226782&
$Conv1D_7_256/StatefulPartitionedCallð
$Conv1D_5_256/StatefulPartitionedCallStatefulPartitionedCall1Batch_Norm_5_128/StatefulPartitionedCall:output:0conv1d_5_256_1916623399conv1d_5_256_1916623401*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_Conv1D_5_256_layer_call_and_return_conditional_losses_19166227052&
$Conv1D_5_256/StatefulPartitionedCallð
$Conv1D_3_256/StatefulPartitionedCallStatefulPartitionedCall1Batch_Norm_3_128/StatefulPartitionedCall:output:0conv1d_3_256_1916623404conv1d_3_256_1916623406*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_Conv1D_3_256_layer_call_and_return_conditional_losses_19166227322&
$Conv1D_3_256/StatefulPartitionedCall¾
(Batch_Norm_7_256/StatefulPartitionedCallStatefulPartitionedCall-Conv1D_7_256/StatefulPartitionedCall:output:0batch_norm_7_256_1916623409batch_norm_7_256_1916623411batch_norm_7_256_1916623413batch_norm_7_256_1916623415*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_Batch_Norm_7_256_layer_call_and_return_conditional_losses_19166223582*
(Batch_Norm_7_256/StatefulPartitionedCall¾
(Batch_Norm_5_256/StatefulPartitionedCallStatefulPartitionedCall-Conv1D_5_256/StatefulPartitionedCall:output:0batch_norm_5_256_1916623418batch_norm_5_256_1916623420batch_norm_5_256_1916623422batch_norm_5_256_1916623424*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_Batch_Norm_5_256_layer_call_and_return_conditional_losses_19166221962*
(Batch_Norm_5_256/StatefulPartitionedCall¾
(Batch_Norm_3_256/StatefulPartitionedCallStatefulPartitionedCall-Conv1D_3_256/StatefulPartitionedCall:output:0batch_norm_3_256_1916623427batch_norm_3_256_1916623429batch_norm_3_256_1916623431batch_norm_3_256_1916623433*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_Batch_Norm_3_256_layer_call_and_return_conditional_losses_19166220342*
(Batch_Norm_3_256/StatefulPartitionedCall
Max_Pool_3/PartitionedCallPartitionedCall1Batch_Norm_3_256/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Max_Pool_3_layer_call_and_return_conditional_losses_19166225032
Max_Pool_3/PartitionedCall
Max_Pool_5/PartitionedCallPartitionedCall1Batch_Norm_5_256/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Max_Pool_5_layer_call_and_return_conditional_losses_19166225162
Max_Pool_5/PartitionedCall
Max_Pool_7/PartitionedCallPartitionedCall1Batch_Norm_7_256/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Max_Pool_7_layer_call_and_return_conditional_losses_19166225292
Max_Pool_7/PartitionedCallÝ
 Pool_Concatenate/PartitionedCallPartitionedCall#Max_Pool_3/PartitionedCall:output:0#Max_Pool_5/PartitionedCall:output:0#Max_Pool_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_Pool_Concatenate_layer_call_and_return_conditional_losses_19166227762"
 Pool_Concatenate/PartitionedCall 
#Concatenate_Dropout/PartitionedCallPartitionedCall)Pool_Concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_Concatenate_Dropout_layer_call_and_return_conditional_losses_19166227832%
#Concatenate_Dropout/PartitionedCallÓ
"Prediction/StatefulPartitionedCallStatefulPartitionedCall,Concatenate_Dropout/PartitionedCall:output:0prediction_1916623441prediction_1916623443*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Prediction_layer_call_and_return_conditional_losses_19166227962$
"Prediction/StatefulPartitionedCall
IdentityIdentity+Prediction/StatefulPartitionedCall:output:0)^Batch_Norm_3_128/StatefulPartitionedCall)^Batch_Norm_3_256/StatefulPartitionedCall)^Batch_Norm_5_128/StatefulPartitionedCall)^Batch_Norm_5_256/StatefulPartitionedCall)^Batch_Norm_7_128/StatefulPartitionedCall)^Batch_Norm_7_256/StatefulPartitionedCall%^Conv1D_3_128/StatefulPartitionedCall%^Conv1D_3_256/StatefulPartitionedCall%^Conv1D_5_128/StatefulPartitionedCall%^Conv1D_5_256/StatefulPartitionedCall%^Conv1D_7_128/StatefulPartitionedCall%^Conv1D_7_256/StatefulPartitionedCall#^Prediction/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapeso
m:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2T
(Batch_Norm_3_128/StatefulPartitionedCall(Batch_Norm_3_128/StatefulPartitionedCall2T
(Batch_Norm_3_256/StatefulPartitionedCall(Batch_Norm_3_256/StatefulPartitionedCall2T
(Batch_Norm_5_128/StatefulPartitionedCall(Batch_Norm_5_128/StatefulPartitionedCall2T
(Batch_Norm_5_256/StatefulPartitionedCall(Batch_Norm_5_256/StatefulPartitionedCall2T
(Batch_Norm_7_128/StatefulPartitionedCall(Batch_Norm_7_128/StatefulPartitionedCall2T
(Batch_Norm_7_256/StatefulPartitionedCall(Batch_Norm_7_256/StatefulPartitionedCall2L
$Conv1D_3_128/StatefulPartitionedCall$Conv1D_3_128/StatefulPartitionedCall2L
$Conv1D_3_256/StatefulPartitionedCall$Conv1D_3_256/StatefulPartitionedCall2L
$Conv1D_5_128/StatefulPartitionedCall$Conv1D_5_128/StatefulPartitionedCall2L
$Conv1D_5_256/StatefulPartitionedCall$Conv1D_5_256/StatefulPartitionedCall2L
$Conv1D_7_128/StatefulPartitionedCall$Conv1D_7_128/StatefulPartitionedCall2L
$Conv1D_7_256/StatefulPartitionedCall$Conv1D_7_256/StatefulPartitionedCall2H
"Prediction/StatefulPartitionedCall"Prediction/StatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameInputs
ñ
³
P__inference_Batch_Norm_3_128_layer_call_and_return_conditional_losses_1916624490

inputs0
!batchnorm_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	2
#batchnorm_readvariableop_1_resource:	2
#batchnorm_readvariableop_2_resource:	
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1é
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü*
í
P__inference_Batch_Norm_3_128_layer_call_and_return_conditional_losses_1916624524

inputs6
'assignmovingavg_readvariableop_resource:	8
)assignmovingavg_1_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	0
!batchnorm_readvariableop_resource:	
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:2
moments/StopGradient²
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices·
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay¥
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg/mul¿
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay«
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp¡
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
©
f
J__inference_Max_Pool_5_layer_call_and_return_conditional_losses_1916622516

inputs
identityp
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Max/reduction_indicest
MaxMaxinputsMax/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Maxi
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¼
Ô
5__inference_Batch_Norm_7_128_layer_call_fn_1916624630

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_Batch_Norm_7_128_layer_call_and_return_conditional_losses_19166219322
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Â
r
S__inference_Concatenate_Dropout_layer_call_and_return_conditional_losses_1916625056

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeµ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
dropout/GreaterEqual/y¿
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü
 
L__inference_Conv1D_3_128_layer_call_and_return_conditional_losses_1916622624

inputsC
+conv1d_expanddims_1_readvariableop_resource:.
biasadd_readvariableop_resource:	

identity_1¢BiasAdd/ReadVariableOp¢"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims/dim 
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
conv1d/ExpandDimsº
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim¹
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:2
conv1d/ExpandDims_1À
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAddo
SigmoidSigmoidBiasAdd:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
Sigmoidp
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
muli
IdentityIdentitymul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

IdentityÕ
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1916622617*V
_output_shapesD
B:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	IdentityN¶

Identity_1IdentityIdentityN:output:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í
¥
1__inference_Conv1D_7_128_layer_call_fn_1916624423

inputs
unknown:
	unknown_0:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_Conv1D_7_128_layer_call_and_return_conditional_losses_19166225702
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ïn
Û
L__inference_Tree_Encoder_layer_call_and_return_conditional_losses_1916623188

inputs/
conv1d_7_128_1916623093:&
conv1d_7_128_1916623095:	/
conv1d_5_128_1916623098:&
conv1d_5_128_1916623100:	/
conv1d_3_128_1916623103:&
conv1d_3_128_1916623105:	*
batch_norm_7_128_1916623108:	*
batch_norm_7_128_1916623110:	*
batch_norm_7_128_1916623112:	*
batch_norm_7_128_1916623114:	*
batch_norm_5_128_1916623117:	*
batch_norm_5_128_1916623119:	*
batch_norm_5_128_1916623121:	*
batch_norm_5_128_1916623123:	*
batch_norm_3_128_1916623126:	*
batch_norm_3_128_1916623128:	*
batch_norm_3_128_1916623130:	*
batch_norm_3_128_1916623132:	/
conv1d_7_256_1916623135:&
conv1d_7_256_1916623137:	/
conv1d_5_256_1916623140:&
conv1d_5_256_1916623142:	/
conv1d_3_256_1916623145:&
conv1d_3_256_1916623147:	*
batch_norm_7_256_1916623150:	*
batch_norm_7_256_1916623152:	*
batch_norm_7_256_1916623154:	*
batch_norm_7_256_1916623156:	*
batch_norm_5_256_1916623159:	*
batch_norm_5_256_1916623161:	*
batch_norm_5_256_1916623163:	*
batch_norm_5_256_1916623165:	*
batch_norm_3_256_1916623168:	*
batch_norm_3_256_1916623170:	*
batch_norm_3_256_1916623172:	*
batch_norm_3_256_1916623174:	(
prediction_1916623182:	#
prediction_1916623184:
identity¢(Batch_Norm_3_128/StatefulPartitionedCall¢(Batch_Norm_3_256/StatefulPartitionedCall¢(Batch_Norm_5_128/StatefulPartitionedCall¢(Batch_Norm_5_256/StatefulPartitionedCall¢(Batch_Norm_7_128/StatefulPartitionedCall¢(Batch_Norm_7_256/StatefulPartitionedCall¢+Concatenate_Dropout/StatefulPartitionedCall¢$Conv1D_3_128/StatefulPartitionedCall¢$Conv1D_3_256/StatefulPartitionedCall¢$Conv1D_5_128/StatefulPartitionedCall¢$Conv1D_5_256/StatefulPartitionedCall¢$Conv1D_7_128/StatefulPartitionedCall¢$Conv1D_7_256/StatefulPartitionedCall¢)Embedding_Dropout/StatefulPartitionedCall¢"Prediction/StatefulPartitionedCall
)Embedding_Dropout/StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_Embedding_Dropout_layer_call_and_return_conditional_losses_19166230032+
)Embedding_Dropout/StatefulPartitionedCallñ
$Conv1D_7_128/StatefulPartitionedCallStatefulPartitionedCall2Embedding_Dropout/StatefulPartitionedCall:output:0conv1d_7_128_1916623093conv1d_7_128_1916623095*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_Conv1D_7_128_layer_call_and_return_conditional_losses_19166225702&
$Conv1D_7_128/StatefulPartitionedCallñ
$Conv1D_5_128/StatefulPartitionedCallStatefulPartitionedCall2Embedding_Dropout/StatefulPartitionedCall:output:0conv1d_5_128_1916623098conv1d_5_128_1916623100*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_Conv1D_5_128_layer_call_and_return_conditional_losses_19166225972&
$Conv1D_5_128/StatefulPartitionedCallñ
$Conv1D_3_128/StatefulPartitionedCallStatefulPartitionedCall2Embedding_Dropout/StatefulPartitionedCall:output:0conv1d_3_128_1916623103conv1d_3_128_1916623105*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_Conv1D_3_128_layer_call_and_return_conditional_losses_19166226242&
$Conv1D_3_128/StatefulPartitionedCall¼
(Batch_Norm_7_128/StatefulPartitionedCallStatefulPartitionedCall-Conv1D_7_128/StatefulPartitionedCall:output:0batch_norm_7_128_1916623108batch_norm_7_128_1916623110batch_norm_7_128_1916623112batch_norm_7_128_1916623114*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_Batch_Norm_7_128_layer_call_and_return_conditional_losses_19166219322*
(Batch_Norm_7_128/StatefulPartitionedCall¼
(Batch_Norm_5_128/StatefulPartitionedCallStatefulPartitionedCall-Conv1D_5_128/StatefulPartitionedCall:output:0batch_norm_5_128_1916623117batch_norm_5_128_1916623119batch_norm_5_128_1916623121batch_norm_5_128_1916623123*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_Batch_Norm_5_128_layer_call_and_return_conditional_losses_19166217702*
(Batch_Norm_5_128/StatefulPartitionedCall¼
(Batch_Norm_3_128/StatefulPartitionedCallStatefulPartitionedCall-Conv1D_3_128/StatefulPartitionedCall:output:0batch_norm_3_128_1916623126batch_norm_3_128_1916623128batch_norm_3_128_1916623130batch_norm_3_128_1916623132*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_Batch_Norm_3_128_layer_call_and_return_conditional_losses_19166216082*
(Batch_Norm_3_128/StatefulPartitionedCallð
$Conv1D_7_256/StatefulPartitionedCallStatefulPartitionedCall1Batch_Norm_7_128/StatefulPartitionedCall:output:0conv1d_7_256_1916623135conv1d_7_256_1916623137*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_Conv1D_7_256_layer_call_and_return_conditional_losses_19166226782&
$Conv1D_7_256/StatefulPartitionedCallð
$Conv1D_5_256/StatefulPartitionedCallStatefulPartitionedCall1Batch_Norm_5_128/StatefulPartitionedCall:output:0conv1d_5_256_1916623140conv1d_5_256_1916623142*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_Conv1D_5_256_layer_call_and_return_conditional_losses_19166227052&
$Conv1D_5_256/StatefulPartitionedCallð
$Conv1D_3_256/StatefulPartitionedCallStatefulPartitionedCall1Batch_Norm_3_128/StatefulPartitionedCall:output:0conv1d_3_256_1916623145conv1d_3_256_1916623147*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_Conv1D_3_256_layer_call_and_return_conditional_losses_19166227322&
$Conv1D_3_256/StatefulPartitionedCall¼
(Batch_Norm_7_256/StatefulPartitionedCallStatefulPartitionedCall-Conv1D_7_256/StatefulPartitionedCall:output:0batch_norm_7_256_1916623150batch_norm_7_256_1916623152batch_norm_7_256_1916623154batch_norm_7_256_1916623156*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_Batch_Norm_7_256_layer_call_and_return_conditional_losses_19166224182*
(Batch_Norm_7_256/StatefulPartitionedCall¼
(Batch_Norm_5_256/StatefulPartitionedCallStatefulPartitionedCall-Conv1D_5_256/StatefulPartitionedCall:output:0batch_norm_5_256_1916623159batch_norm_5_256_1916623161batch_norm_5_256_1916623163batch_norm_5_256_1916623165*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_Batch_Norm_5_256_layer_call_and_return_conditional_losses_19166222562*
(Batch_Norm_5_256/StatefulPartitionedCall¼
(Batch_Norm_3_256/StatefulPartitionedCallStatefulPartitionedCall-Conv1D_3_256/StatefulPartitionedCall:output:0batch_norm_3_256_1916623168batch_norm_3_256_1916623170batch_norm_3_256_1916623172batch_norm_3_256_1916623174*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_Batch_Norm_3_256_layer_call_and_return_conditional_losses_19166220942*
(Batch_Norm_3_256/StatefulPartitionedCall
Max_Pool_3/PartitionedCallPartitionedCall1Batch_Norm_3_256/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Max_Pool_3_layer_call_and_return_conditional_losses_19166225032
Max_Pool_3/PartitionedCall
Max_Pool_5/PartitionedCallPartitionedCall1Batch_Norm_5_256/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Max_Pool_5_layer_call_and_return_conditional_losses_19166225162
Max_Pool_5/PartitionedCall
Max_Pool_7/PartitionedCallPartitionedCall1Batch_Norm_7_256/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Max_Pool_7_layer_call_and_return_conditional_losses_19166225292
Max_Pool_7/PartitionedCallÝ
 Pool_Concatenate/PartitionedCallPartitionedCall#Max_Pool_3/PartitionedCall:output:0#Max_Pool_5/PartitionedCall:output:0#Max_Pool_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_Pool_Concatenate_layer_call_and_return_conditional_losses_19166227762"
 Pool_Concatenate/PartitionedCallä
+Concatenate_Dropout/StatefulPartitionedCallStatefulPartitionedCall)Pool_Concatenate/PartitionedCall:output:0*^Embedding_Dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_Concatenate_Dropout_layer_call_and_return_conditional_losses_19166229122-
+Concatenate_Dropout/StatefulPartitionedCallÛ
"Prediction/StatefulPartitionedCallStatefulPartitionedCall4Concatenate_Dropout/StatefulPartitionedCall:output:0prediction_1916623182prediction_1916623184*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Prediction_layer_call_and_return_conditional_losses_19166227962$
"Prediction/StatefulPartitionedCallê
IdentityIdentity+Prediction/StatefulPartitionedCall:output:0)^Batch_Norm_3_128/StatefulPartitionedCall)^Batch_Norm_3_256/StatefulPartitionedCall)^Batch_Norm_5_128/StatefulPartitionedCall)^Batch_Norm_5_256/StatefulPartitionedCall)^Batch_Norm_7_128/StatefulPartitionedCall)^Batch_Norm_7_256/StatefulPartitionedCall,^Concatenate_Dropout/StatefulPartitionedCall%^Conv1D_3_128/StatefulPartitionedCall%^Conv1D_3_256/StatefulPartitionedCall%^Conv1D_5_128/StatefulPartitionedCall%^Conv1D_5_256/StatefulPartitionedCall%^Conv1D_7_128/StatefulPartitionedCall%^Conv1D_7_256/StatefulPartitionedCall*^Embedding_Dropout/StatefulPartitionedCall#^Prediction/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapeso
m:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2T
(Batch_Norm_3_128/StatefulPartitionedCall(Batch_Norm_3_128/StatefulPartitionedCall2T
(Batch_Norm_3_256/StatefulPartitionedCall(Batch_Norm_3_256/StatefulPartitionedCall2T
(Batch_Norm_5_128/StatefulPartitionedCall(Batch_Norm_5_128/StatefulPartitionedCall2T
(Batch_Norm_5_256/StatefulPartitionedCall(Batch_Norm_5_256/StatefulPartitionedCall2T
(Batch_Norm_7_128/StatefulPartitionedCall(Batch_Norm_7_128/StatefulPartitionedCall2T
(Batch_Norm_7_256/StatefulPartitionedCall(Batch_Norm_7_256/StatefulPartitionedCall2Z
+Concatenate_Dropout/StatefulPartitionedCall+Concatenate_Dropout/StatefulPartitionedCall2L
$Conv1D_3_128/StatefulPartitionedCall$Conv1D_3_128/StatefulPartitionedCall2L
$Conv1D_3_256/StatefulPartitionedCall$Conv1D_3_256/StatefulPartitionedCall2L
$Conv1D_5_128/StatefulPartitionedCall$Conv1D_5_128/StatefulPartitionedCall2L
$Conv1D_5_256/StatefulPartitionedCall$Conv1D_5_256/StatefulPartitionedCall2L
$Conv1D_7_128/StatefulPartitionedCall$Conv1D_7_128/StatefulPartitionedCall2L
$Conv1D_7_256/StatefulPartitionedCall$Conv1D_7_256/StatefulPartitionedCall2V
)Embedding_Dropout/StatefulPartitionedCall)Embedding_Dropout/StatefulPartitionedCall2H
"Prediction/StatefulPartitionedCall"Prediction/StatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü*
í
P__inference_Batch_Norm_5_128_layer_call_and_return_conditional_losses_1916624604

inputs6
'assignmovingavg_readvariableop_resource:	8
)assignmovingavg_1_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	0
!batchnorm_readvariableop_resource:	
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:*
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:2
moments/StopGradient²
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices·
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay¥
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg/mul¿
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay«
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp¡
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"ÌL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¹
serving_default¥
G
Inputs=
serving_default_Inputs:0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ>

Prediction0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:°°
¾
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer_with_weights-7

layer-9
layer_with_weights-8
layer-10
layer_with_weights-9
layer-11
layer_with_weights-10
layer-12
layer_with_weights-11
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer_with_weights-12
layer-19

signatures
#_self_saveable_object_factories
	optimizer
loss
trainable_variables
regularization_losses
	variables
	keras_api
__call__
+&call_and_return_all_conditional_losses
_default_save_signature"··
_tf_keras_network·{"name": "Tree_Encoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "Tree_Encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Inputs"}, "name": "Inputs", "inbound_nodes": []}, {"class_name": "Dropout", "config": {"name": "Embedding_Dropout", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "Embedding_Dropout", "inbound_nodes": [[["Inputs", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "Conv1D_3_128", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_3_128", "inbound_nodes": [[["Embedding_Dropout", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "Conv1D_5_128", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_5_128", "inbound_nodes": [[["Embedding_Dropout", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "Conv1D_7_128", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [7]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_7_128", "inbound_nodes": [[["Embedding_Dropout", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "Batch_Norm_3_128", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "Batch_Norm_3_128", "inbound_nodes": [[["Conv1D_3_128", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "Batch_Norm_5_128", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "Batch_Norm_5_128", "inbound_nodes": [[["Conv1D_5_128", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "Batch_Norm_7_128", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "Batch_Norm_7_128", "inbound_nodes": [[["Conv1D_7_128", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "Conv1D_3_256", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_3_256", "inbound_nodes": [[["Batch_Norm_3_128", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "Conv1D_5_256", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_5_256", "inbound_nodes": [[["Batch_Norm_5_128", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "Conv1D_7_256", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [7]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_7_256", "inbound_nodes": [[["Batch_Norm_7_128", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "Batch_Norm_3_256", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "Batch_Norm_3_256", "inbound_nodes": [[["Conv1D_3_256", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "Batch_Norm_5_256", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "Batch_Norm_5_256", "inbound_nodes": [[["Conv1D_5_256", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "Batch_Norm_7_256", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "Batch_Norm_7_256", "inbound_nodes": [[["Conv1D_7_256", 0, 0, {}]]]}, {"class_name": "GlobalMaxPooling1D", "config": {"name": "Max_Pool_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "Max_Pool_3", "inbound_nodes": [[["Batch_Norm_3_256", 0, 0, {}]]]}, {"class_name": "GlobalMaxPooling1D", "config": {"name": "Max_Pool_5", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "Max_Pool_5", "inbound_nodes": [[["Batch_Norm_5_256", 0, 0, {}]]]}, {"class_name": "GlobalMaxPooling1D", "config": {"name": "Max_Pool_7", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "Max_Pool_7", "inbound_nodes": [[["Batch_Norm_7_256", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "Pool_Concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "Pool_Concatenate", "inbound_nodes": [[["Max_Pool_3", 0, 0, {}], ["Max_Pool_5", 0, 0, {}], ["Max_Pool_7", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "Concatenate_Dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "Concatenate_Dropout", "inbound_nodes": [[["Pool_Concatenate", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Prediction", "trainable": true, "dtype": "float32", "units": 28, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Prediction", "inbound_nodes": [[["Concatenate_Dropout", 0, 0, {}]]]}], "input_layers": [["Inputs", 0, 0]], "output_layers": [["Prediction", 0, 0]]}, "shared_object_id": 58, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 128]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 128]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, null, 128]}, "float32", "Inputs"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "Tree_Encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Inputs"}, "name": "Inputs", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Dropout", "config": {"name": "Embedding_Dropout", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "Embedding_Dropout", "inbound_nodes": [[["Inputs", 0, 0, {}]]], "shared_object_id": 1}, {"class_name": "Conv1D", "config": {"name": "Conv1D_3_128", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 2}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 3}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_3_128", "inbound_nodes": [[["Embedding_Dropout", 0, 0, {}]]], "shared_object_id": 4}, {"class_name": "Conv1D", "config": {"name": "Conv1D_5_128", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_5_128", "inbound_nodes": [[["Embedding_Dropout", 0, 0, {}]]], "shared_object_id": 7}, {"class_name": "Conv1D", "config": {"name": "Conv1D_7_128", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [7]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_7_128", "inbound_nodes": [[["Embedding_Dropout", 0, 0, {}]]], "shared_object_id": 10}, {"class_name": "BatchNormalization", "config": {"name": "Batch_Norm_3_128", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 12}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 14}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "Batch_Norm_3_128", "inbound_nodes": [[["Conv1D_3_128", 0, 0, {}]]], "shared_object_id": 15}, {"class_name": "BatchNormalization", "config": {"name": "Batch_Norm_5_128", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 17}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 18}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 19}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "Batch_Norm_5_128", "inbound_nodes": [[["Conv1D_5_128", 0, 0, {}]]], "shared_object_id": 20}, {"class_name": "BatchNormalization", "config": {"name": "Batch_Norm_7_128", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 22}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 24}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "Batch_Norm_7_128", "inbound_nodes": [[["Conv1D_7_128", 0, 0, {}]]], "shared_object_id": 25}, {"class_name": "Conv1D", "config": {"name": "Conv1D_3_256", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 26}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 27}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_3_256", "inbound_nodes": [[["Batch_Norm_3_128", 0, 0, {}]]], "shared_object_id": 28}, {"class_name": "Conv1D", "config": {"name": "Conv1D_5_256", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 29}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 30}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_5_256", "inbound_nodes": [[["Batch_Norm_5_128", 0, 0, {}]]], "shared_object_id": 31}, {"class_name": "Conv1D", "config": {"name": "Conv1D_7_256", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [7]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 32}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 33}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_7_256", "inbound_nodes": [[["Batch_Norm_7_128", 0, 0, {}]]], "shared_object_id": 34}, {"class_name": "BatchNormalization", "config": {"name": "Batch_Norm_3_256", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 35}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 36}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 37}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 38}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "Batch_Norm_3_256", "inbound_nodes": [[["Conv1D_3_256", 0, 0, {}]]], "shared_object_id": 39}, {"class_name": "BatchNormalization", "config": {"name": "Batch_Norm_5_256", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 40}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 41}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 42}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 43}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "Batch_Norm_5_256", "inbound_nodes": [[["Conv1D_5_256", 0, 0, {}]]], "shared_object_id": 44}, {"class_name": "BatchNormalization", "config": {"name": "Batch_Norm_7_256", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 45}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 46}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 47}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 48}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "Batch_Norm_7_256", "inbound_nodes": [[["Conv1D_7_256", 0, 0, {}]]], "shared_object_id": 49}, {"class_name": "GlobalMaxPooling1D", "config": {"name": "Max_Pool_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "Max_Pool_3", "inbound_nodes": [[["Batch_Norm_3_256", 0, 0, {}]]], "shared_object_id": 50}, {"class_name": "GlobalMaxPooling1D", "config": {"name": "Max_Pool_5", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "Max_Pool_5", "inbound_nodes": [[["Batch_Norm_5_256", 0, 0, {}]]], "shared_object_id": 51}, {"class_name": "GlobalMaxPooling1D", "config": {"name": "Max_Pool_7", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "Max_Pool_7", "inbound_nodes": [[["Batch_Norm_7_256", 0, 0, {}]]], "shared_object_id": 52}, {"class_name": "Concatenate", "config": {"name": "Pool_Concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "Pool_Concatenate", "inbound_nodes": [[["Max_Pool_3", 0, 0, {}], ["Max_Pool_5", 0, 0, {}], ["Max_Pool_7", 0, 0, {}]]], "shared_object_id": 53}, {"class_name": "Dropout", "config": {"name": "Concatenate_Dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "Concatenate_Dropout", "inbound_nodes": [[["Pool_Concatenate", 0, 0, {}]]], "shared_object_id": 54}, {"class_name": "Dense", "config": {"name": "Prediction", "trainable": true, "dtype": "float32", "units": 28, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 55}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 56}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Prediction", "inbound_nodes": [[["Concatenate_Dropout", 0, 0, {}]]], "shared_object_id": 57}], "input_layers": [["Inputs", 0, 0]], "output_layers": [["Prediction", 0, 0]]}}, "training_config": {"loss": null, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 0.001, "decay": 0.0, "rho": 0.9, "momentum": 0.0, "epsilon": 1e-07, "centered": false}}}}

#_self_saveable_object_factories"ô
_tf_keras_input_layerÔ{"class_name": "InputLayer", "name": "Inputs", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 128]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Inputs"}}
Þ
#_self_saveable_object_factories
trainable_variables
 regularization_losses
!	variables
"	keras_api
__call__
+&call_and_return_all_conditional_losses"¨
_tf_keras_layer{"name": "Embedding_Dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "Embedding_Dropout", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "inbound_nodes": [[["Inputs", 0, 0, {}]]], "shared_object_id": 1}
­

#kernel
$bias
#%_self_saveable_object_factories
&trainable_variables
'regularization_losses
(	variables
)	keras_api
__call__
+&call_and_return_all_conditional_losses"á	
_tf_keras_layerÇ	{"name": "Conv1D_3_128", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "Conv1D_3_128", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 2}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 3}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["Embedding_Dropout", 0, 0, {}]]], "shared_object_id": 4, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 128}}, "shared_object_id": 60}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 128]}}
­

*kernel
+bias
#,_self_saveable_object_factories
-trainable_variables
.regularization_losses
/	variables
0	keras_api
__call__
+&call_and_return_all_conditional_losses"á	
_tf_keras_layerÇ	{"name": "Conv1D_5_128", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "Conv1D_5_128", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["Embedding_Dropout", 0, 0, {}]]], "shared_object_id": 7, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 128}}, "shared_object_id": 61}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 128]}}
®

1kernel
2bias
#3_self_saveable_object_factories
4trainable_variables
5regularization_losses
6	variables
7	keras_api
__call__
+&call_and_return_all_conditional_losses"â	
_tf_keras_layerÈ	{"name": "Conv1D_7_128", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "Conv1D_7_128", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [7]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["Embedding_Dropout", 0, 0, {}]]], "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 128}}, "shared_object_id": 62}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 128]}}

8axis
	9gamma
:beta
;moving_mean
<moving_variance
#=_self_saveable_object_factories
>trainable_variables
?regularization_losses
@	variables
A	keras_api
__call__
+&call_and_return_all_conditional_losses"	
_tf_keras_layer	{"name": "Batch_Norm_3_128", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "Batch_Norm_3_128", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 12}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 14}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["Conv1D_3_128", 0, 0, {}]]], "shared_object_id": 15, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 128}}, "shared_object_id": 63}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 128]}}

Baxis
	Cgamma
Dbeta
Emoving_mean
Fmoving_variance
#G_self_saveable_object_factories
Htrainable_variables
Iregularization_losses
J	variables
K	keras_api
__call__
+ &call_and_return_all_conditional_losses"	
_tf_keras_layer	{"name": "Batch_Norm_5_128", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "Batch_Norm_5_128", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 17}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 18}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 19}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["Conv1D_5_128", 0, 0, {}]]], "shared_object_id": 20, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 128}}, "shared_object_id": 64}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 128]}}

Laxis
	Mgamma
Nbeta
Omoving_mean
Pmoving_variance
#Q_self_saveable_object_factories
Rtrainable_variables
Sregularization_losses
T	variables
U	keras_api
¡__call__
+¢&call_and_return_all_conditional_losses"	
_tf_keras_layer	{"name": "Batch_Norm_7_128", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "Batch_Norm_7_128", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 22}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 24}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["Conv1D_7_128", 0, 0, {}]]], "shared_object_id": 25, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 128}}, "shared_object_id": 65}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 128]}}
¯

Vkernel
Wbias
#X_self_saveable_object_factories
Ytrainable_variables
Zregularization_losses
[	variables
\	keras_api
£__call__
+¤&call_and_return_all_conditional_losses"ã	
_tf_keras_layerÉ	{"name": "Conv1D_3_256", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "Conv1D_3_256", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 26}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 27}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["Batch_Norm_3_128", 0, 0, {}]]], "shared_object_id": 28, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 128}}, "shared_object_id": 66}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 128]}}
¯

]kernel
^bias
#__self_saveable_object_factories
`trainable_variables
aregularization_losses
b	variables
c	keras_api
¥__call__
+¦&call_and_return_all_conditional_losses"ã	
_tf_keras_layerÉ	{"name": "Conv1D_5_256", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "Conv1D_5_256", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 29}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 30}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["Batch_Norm_5_128", 0, 0, {}]]], "shared_object_id": 31, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 128}}, "shared_object_id": 67}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 128]}}
¯

dkernel
ebias
#f_self_saveable_object_factories
gtrainable_variables
hregularization_losses
i	variables
j	keras_api
§__call__
+¨&call_and_return_all_conditional_losses"ã	
_tf_keras_layerÉ	{"name": "Conv1D_7_256", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "Conv1D_7_256", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [7]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 32}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 33}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["Batch_Norm_7_128", 0, 0, {}]]], "shared_object_id": 34, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 128}}, "shared_object_id": 68}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 128]}}

kaxis
	lgamma
mbeta
nmoving_mean
omoving_variance
#p_self_saveable_object_factories
qtrainable_variables
rregularization_losses
s	variables
t	keras_api
©__call__
+ª&call_and_return_all_conditional_losses"	
_tf_keras_layer	{"name": "Batch_Norm_3_256", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "Batch_Norm_3_256", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 35}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 36}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 37}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 38}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["Conv1D_3_256", 0, 0, {}]]], "shared_object_id": 39, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 256}}, "shared_object_id": 69}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 256]}}

uaxis
	vgamma
wbeta
xmoving_mean
ymoving_variance
#z_self_saveable_object_factories
{trainable_variables
|regularization_losses
}	variables
~	keras_api
«__call__
+¬&call_and_return_all_conditional_losses"	
_tf_keras_layer	{"name": "Batch_Norm_5_256", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "Batch_Norm_5_256", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 40}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 41}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 42}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 43}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["Conv1D_5_256", 0, 0, {}]]], "shared_object_id": 44, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 256}}, "shared_object_id": 70}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 256]}}
¡
axis

gamma
	beta
moving_mean
moving_variance
$_self_saveable_object_factories
trainable_variables
regularization_losses
	variables
	keras_api
­__call__
+®&call_and_return_all_conditional_losses"	
_tf_keras_layer	{"name": "Batch_Norm_7_256", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "Batch_Norm_7_256", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 45}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 46}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 47}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 48}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["Conv1D_7_256", 0, 0, {}]]], "shared_object_id": 49, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 256}}, "shared_object_id": 71}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 256]}}

$_self_saveable_object_factories
trainable_variables
regularization_losses
	variables
	keras_api
¯__call__
+°&call_and_return_all_conditional_losses"É
_tf_keras_layer¯{"name": "Max_Pool_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "GlobalMaxPooling1D", "config": {"name": "Max_Pool_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["Batch_Norm_3_256", 0, 0, {}]]], "shared_object_id": 50, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 72}}

$_self_saveable_object_factories
trainable_variables
regularization_losses
	variables
	keras_api
±__call__
+²&call_and_return_all_conditional_losses"É
_tf_keras_layer¯{"name": "Max_Pool_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "GlobalMaxPooling1D", "config": {"name": "Max_Pool_5", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["Batch_Norm_5_256", 0, 0, {}]]], "shared_object_id": 51, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 73}}

$_self_saveable_object_factories
trainable_variables
regularization_losses
	variables
	keras_api
³__call__
+´&call_and_return_all_conditional_losses"É
_tf_keras_layer¯{"name": "Max_Pool_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "GlobalMaxPooling1D", "config": {"name": "Max_Pool_7", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["Batch_Norm_7_256", 0, 0, {}]]], "shared_object_id": 52, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 74}}
±
$_self_saveable_object_factories
trainable_variables
regularization_losses
	variables
	keras_api
µ__call__
+¶&call_and_return_all_conditional_losses"ö
_tf_keras_layerÜ{"name": "Pool_Concatenate", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Concatenate", "config": {"name": "Pool_Concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "inbound_nodes": [[["Max_Pool_3", 0, 0, {}], ["Max_Pool_5", 0, 0, {}], ["Max_Pool_7", 0, 0, {}]]], "shared_object_id": 53, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 256]}, {"class_name": "TensorShape", "items": [null, 256]}, {"class_name": "TensorShape", "items": [null, 256]}]}
ò
$_self_saveable_object_factories
trainable_variables
regularization_losses
 	variables
¡	keras_api
·__call__
+¸&call_and_return_all_conditional_losses"·
_tf_keras_layer{"name": "Concatenate_Dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "Concatenate_Dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "inbound_nodes": [[["Pool_Concatenate", 0, 0, {}]]], "shared_object_id": 54}
Á	
¢kernel
	£bias
$¤_self_saveable_object_factories
¥trainable_variables
¦regularization_losses
§	variables
¨	keras_api
¹__call__
+º&call_and_return_all_conditional_losses"î
_tf_keras_layerÔ{"name": "Prediction", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "Prediction", "trainable": true, "dtype": "float32", "units": 28, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 55}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 56}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["Concatenate_Dropout", 0, 0, {}]]], "shared_object_id": 57, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 768}}, "shared_object_id": 75}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 768]}}
-
»serving_default"
signature_map
 "
trackable_dict_wrapper
"
	optimizer
 "
trackable_dict_wrapper
ê
#0
$1
*2
+3
14
25
96
:7
C8
D9
M10
N11
V12
W13
]14
^15
d16
e17
l18
m19
v20
w21
22
23
¢24
£25"
trackable_list_wrapper
 "
trackable_list_wrapper
Ì
#0
$1
*2
+3
14
25
96
:7
;8
<9
C10
D11
E12
F13
M14
N15
O16
P17
V18
W19
]20
^21
d22
e23
l24
m25
n26
o27
v28
w29
x30
y31
32
33
34
35
¢36
£37"
trackable_list_wrapper
Ó
trainable_variables
©non_trainable_variables
 ªlayer_regularization_losses
regularization_losses
	variables
«metrics
¬layers
­layer_metrics
__call__
_default_save_signature
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
trainable_variables
®non_trainable_variables
 ¯layer_regularization_losses
 regularization_losses
!	variables
°metrics
±layers
²layer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
+:)2Conv1D_3_128/kernel
 :2Conv1D_3_128/bias
 "
trackable_dict_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
µ
&trainable_variables
³non_trainable_variables
 ´layer_regularization_losses
'regularization_losses
(	variables
µmetrics
¶layers
·layer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
+:)2Conv1D_5_128/kernel
 :2Conv1D_5_128/bias
 "
trackable_dict_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
µ
-trainable_variables
¸non_trainable_variables
 ¹layer_regularization_losses
.regularization_losses
/	variables
ºmetrics
»layers
¼layer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
+:)2Conv1D_7_128/kernel
 :2Conv1D_7_128/bias
 "
trackable_dict_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
µ
4trainable_variables
½non_trainable_variables
 ¾layer_regularization_losses
5regularization_losses
6	variables
¿metrics
Àlayers
Álayer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
%:#2Batch_Norm_3_128/gamma
$:"2Batch_Norm_3_128/beta
-:+ (2Batch_Norm_3_128/moving_mean
1:/ (2 Batch_Norm_3_128/moving_variance
 "
trackable_dict_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
90
:1
;2
<3"
trackable_list_wrapper
µ
>trainable_variables
Ânon_trainable_variables
 Ãlayer_regularization_losses
?regularization_losses
@	variables
Ämetrics
Ålayers
Ælayer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
%:#2Batch_Norm_5_128/gamma
$:"2Batch_Norm_5_128/beta
-:+ (2Batch_Norm_5_128/moving_mean
1:/ (2 Batch_Norm_5_128/moving_variance
 "
trackable_dict_wrapper
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
C0
D1
E2
F3"
trackable_list_wrapper
µ
Htrainable_variables
Çnon_trainable_variables
 Èlayer_regularization_losses
Iregularization_losses
J	variables
Émetrics
Êlayers
Ëlayer_metrics
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
%:#2Batch_Norm_7_128/gamma
$:"2Batch_Norm_7_128/beta
-:+ (2Batch_Norm_7_128/moving_mean
1:/ (2 Batch_Norm_7_128/moving_variance
 "
trackable_dict_wrapper
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
M0
N1
O2
P3"
trackable_list_wrapper
µ
Rtrainable_variables
Ìnon_trainable_variables
 Ílayer_regularization_losses
Sregularization_losses
T	variables
Îmetrics
Ïlayers
Ðlayer_metrics
¡__call__
+¢&call_and_return_all_conditional_losses
'¢"call_and_return_conditional_losses"
_generic_user_object
+:)2Conv1D_3_256/kernel
 :2Conv1D_3_256/bias
 "
trackable_dict_wrapper
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
µ
Ytrainable_variables
Ñnon_trainable_variables
 Òlayer_regularization_losses
Zregularization_losses
[	variables
Ómetrics
Ôlayers
Õlayer_metrics
£__call__
+¤&call_and_return_all_conditional_losses
'¤"call_and_return_conditional_losses"
_generic_user_object
+:)2Conv1D_5_256/kernel
 :2Conv1D_5_256/bias
 "
trackable_dict_wrapper
.
]0
^1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
µ
`trainable_variables
Önon_trainable_variables
 ×layer_regularization_losses
aregularization_losses
b	variables
Ømetrics
Ùlayers
Úlayer_metrics
¥__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses"
_generic_user_object
+:)2Conv1D_7_256/kernel
 :2Conv1D_7_256/bias
 "
trackable_dict_wrapper
.
d0
e1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
µ
gtrainable_variables
Ûnon_trainable_variables
 Ülayer_regularization_losses
hregularization_losses
i	variables
Ýmetrics
Þlayers
ßlayer_metrics
§__call__
+¨&call_and_return_all_conditional_losses
'¨"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
%:#2Batch_Norm_3_256/gamma
$:"2Batch_Norm_3_256/beta
-:+ (2Batch_Norm_3_256/moving_mean
1:/ (2 Batch_Norm_3_256/moving_variance
 "
trackable_dict_wrapper
.
l0
m1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
l0
m1
n2
o3"
trackable_list_wrapper
µ
qtrainable_variables
ànon_trainable_variables
 álayer_regularization_losses
rregularization_losses
s	variables
âmetrics
ãlayers
älayer_metrics
©__call__
+ª&call_and_return_all_conditional_losses
'ª"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
%:#2Batch_Norm_5_256/gamma
$:"2Batch_Norm_5_256/beta
-:+ (2Batch_Norm_5_256/moving_mean
1:/ (2 Batch_Norm_5_256/moving_variance
 "
trackable_dict_wrapper
.
v0
w1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
v0
w1
x2
y3"
trackable_list_wrapper
µ
{trainable_variables
ånon_trainable_variables
 ælayer_regularization_losses
|regularization_losses
}	variables
çmetrics
èlayers
élayer_metrics
«__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
%:#2Batch_Norm_7_256/gamma
$:"2Batch_Norm_7_256/beta
-:+ (2Batch_Norm_7_256/moving_mean
1:/ (2 Batch_Norm_7_256/moving_variance
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
@
0
1
2
3"
trackable_list_wrapper
¸
trainable_variables
ênon_trainable_variables
 ëlayer_regularization_losses
regularization_losses
	variables
ìmetrics
ílayers
îlayer_metrics
­__call__
+®&call_and_return_all_conditional_losses
'®"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
trainable_variables
ïnon_trainable_variables
 ðlayer_regularization_losses
regularization_losses
	variables
ñmetrics
òlayers
ólayer_metrics
¯__call__
+°&call_and_return_all_conditional_losses
'°"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
trainable_variables
ônon_trainable_variables
 õlayer_regularization_losses
regularization_losses
	variables
ömetrics
÷layers
ølayer_metrics
±__call__
+²&call_and_return_all_conditional_losses
'²"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
trainable_variables
ùnon_trainable_variables
 úlayer_regularization_losses
regularization_losses
	variables
ûmetrics
ülayers
ýlayer_metrics
³__call__
+´&call_and_return_all_conditional_losses
'´"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
trainable_variables
þnon_trainable_variables
 ÿlayer_regularization_losses
regularization_losses
	variables
metrics
layers
layer_metrics
µ__call__
+¶&call_and_return_all_conditional_losses
'¶"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
trainable_variables
non_trainable_variables
 layer_regularization_losses
regularization_losses
 	variables
metrics
layers
layer_metrics
·__call__
+¸&call_and_return_all_conditional_losses
'¸"call_and_return_conditional_losses"
_generic_user_object
$:"	2Prediction/kernel
:2Prediction/bias
 "
trackable_dict_wrapper
0
¢0
£1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
¢0
£1"
trackable_list_wrapper
¸
¥trainable_variables
non_trainable_variables
 layer_regularization_losses
¦regularization_losses
§	variables
metrics
layers
layer_metrics
¹__call__
+º&call_and_return_all_conditional_losses
'º"call_and_return_conditional_losses"
_generic_user_object
x
;0
<1
E2
F3
O4
P5
n6
o7
x8
y9
10
11"
trackable_list_wrapper
 "
trackable_list_wrapper
(
0"
trackable_list_wrapper
¶
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
O0
P1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
n0
o1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
x0
y1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Ø

total

count
	variables
	keras_api"
_tf_keras_metric{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 76}
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
2
1__inference_Tree_Encoder_layer_call_fn_1916622882
1__inference_Tree_Encoder_layer_call_fn_1916623710
1__inference_Tree_Encoder_layer_call_fn_1916623791
1__inference_Tree_Encoder_layer_call_fn_1916623348À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
þ2û
L__inference_Tree_Encoder_layer_call_and_return_conditional_losses_1916624010
L__inference_Tree_Encoder_layer_call_and_return_conditional_losses_1916624327
L__inference_Tree_Encoder_layer_call_and_return_conditional_losses_1916623447
L__inference_Tree_Encoder_layer_call_and_return_conditional_losses_1916623546À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ð2í
%__inference__wrapped_model_1916621524Ã
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *3¢0
.+
Inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª2§
6__inference_Embedding_Dropout_layer_call_fn_1916624332
6__inference_Embedding_Dropout_layer_call_fn_1916624337´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
à2Ý
Q__inference_Embedding_Dropout_layer_call_and_return_conditional_losses_1916624342
Q__inference_Embedding_Dropout_layer_call_and_return_conditional_losses_1916624354´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Û2Ø
1__inference_Conv1D_3_128_layer_call_fn_1916624363¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ö2ó
L__inference_Conv1D_3_128_layer_call_and_return_conditional_losses_1916624384¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Û2Ø
1__inference_Conv1D_5_128_layer_call_fn_1916624393¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ö2ó
L__inference_Conv1D_5_128_layer_call_and_return_conditional_losses_1916624414¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Û2Ø
1__inference_Conv1D_7_128_layer_call_fn_1916624423¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ö2ó
L__inference_Conv1D_7_128_layer_call_and_return_conditional_losses_1916624444¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥
5__inference_Batch_Norm_3_128_layer_call_fn_1916624457
5__inference_Batch_Norm_3_128_layer_call_fn_1916624470´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Þ2Û
P__inference_Batch_Norm_3_128_layer_call_and_return_conditional_losses_1916624490
P__inference_Batch_Norm_3_128_layer_call_and_return_conditional_losses_1916624524´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¨2¥
5__inference_Batch_Norm_5_128_layer_call_fn_1916624537
5__inference_Batch_Norm_5_128_layer_call_fn_1916624550´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Þ2Û
P__inference_Batch_Norm_5_128_layer_call_and_return_conditional_losses_1916624570
P__inference_Batch_Norm_5_128_layer_call_and_return_conditional_losses_1916624604´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¨2¥
5__inference_Batch_Norm_7_128_layer_call_fn_1916624617
5__inference_Batch_Norm_7_128_layer_call_fn_1916624630´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Þ2Û
P__inference_Batch_Norm_7_128_layer_call_and_return_conditional_losses_1916624650
P__inference_Batch_Norm_7_128_layer_call_and_return_conditional_losses_1916624684´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Û2Ø
1__inference_Conv1D_3_256_layer_call_fn_1916624693¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ö2ó
L__inference_Conv1D_3_256_layer_call_and_return_conditional_losses_1916624714¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Û2Ø
1__inference_Conv1D_5_256_layer_call_fn_1916624723¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ö2ó
L__inference_Conv1D_5_256_layer_call_and_return_conditional_losses_1916624744¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Û2Ø
1__inference_Conv1D_7_256_layer_call_fn_1916624753¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ö2ó
L__inference_Conv1D_7_256_layer_call_and_return_conditional_losses_1916624774¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥
5__inference_Batch_Norm_3_256_layer_call_fn_1916624787
5__inference_Batch_Norm_3_256_layer_call_fn_1916624800´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Þ2Û
P__inference_Batch_Norm_3_256_layer_call_and_return_conditional_losses_1916624820
P__inference_Batch_Norm_3_256_layer_call_and_return_conditional_losses_1916624854´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¨2¥
5__inference_Batch_Norm_5_256_layer_call_fn_1916624867
5__inference_Batch_Norm_5_256_layer_call_fn_1916624880´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Þ2Û
P__inference_Batch_Norm_5_256_layer_call_and_return_conditional_losses_1916624900
P__inference_Batch_Norm_5_256_layer_call_and_return_conditional_losses_1916624934´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¨2¥
5__inference_Batch_Norm_7_256_layer_call_fn_1916624947
5__inference_Batch_Norm_7_256_layer_call_fn_1916624960´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Þ2Û
P__inference_Batch_Norm_7_256_layer_call_and_return_conditional_losses_1916624980
P__inference_Batch_Norm_7_256_layer_call_and_return_conditional_losses_1916625014´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
/__inference_Max_Pool_3_layer_call_fn_1916622509Ó
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *3¢0
.+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¥2¢
J__inference_Max_Pool_3_layer_call_and_return_conditional_losses_1916622503Ó
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *3¢0
.+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
/__inference_Max_Pool_5_layer_call_fn_1916622522Ó
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *3¢0
.+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¥2¢
J__inference_Max_Pool_5_layer_call_and_return_conditional_losses_1916622516Ó
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *3¢0
.+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
/__inference_Max_Pool_7_layer_call_fn_1916622535Ó
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *3¢0
.+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¥2¢
J__inference_Max_Pool_7_layer_call_and_return_conditional_losses_1916622529Ó
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *3¢0
.+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ß2Ü
5__inference_Pool_Concatenate_layer_call_fn_1916625021¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_Pool_Concatenate_layer_call_and_return_conditional_losses_1916625029¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
®2«
8__inference_Concatenate_Dropout_layer_call_fn_1916625034
8__inference_Concatenate_Dropout_layer_call_fn_1916625039´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ä2á
S__inference_Concatenate_Dropout_layer_call_and_return_conditional_losses_1916625044
S__inference_Concatenate_Dropout_layer_call_and_return_conditional_losses_1916625056´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ù2Ö
/__inference_Prediction_layer_call_fn_1916625065¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ô2ñ
J__inference_Prediction_layer_call_and_return_conditional_losses_1916625076¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÎBË
(__inference_signature_wrapper_1916623629Inputs"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 Ò
P__inference_Batch_Norm_3_128_layer_call_and_return_conditional_losses_1916624490~<9;:A¢>
7¢4
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ò
P__inference_Batch_Norm_3_128_layer_call_and_return_conditional_losses_1916624524~;<9:A¢>
7¢4
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ª
5__inference_Batch_Norm_3_128_layer_call_fn_1916624457q<9;:A¢>
7¢4
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª
5__inference_Batch_Norm_3_128_layer_call_fn_1916624470q;<9:A¢>
7¢4
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÒ
P__inference_Batch_Norm_3_256_layer_call_and_return_conditional_losses_1916624820~olnmA¢>
7¢4
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ò
P__inference_Batch_Norm_3_256_layer_call_and_return_conditional_losses_1916624854~nolmA¢>
7¢4
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ª
5__inference_Batch_Norm_3_256_layer_call_fn_1916624787qolnmA¢>
7¢4
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª
5__inference_Batch_Norm_3_256_layer_call_fn_1916624800qnolmA¢>
7¢4
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÒ
P__inference_Batch_Norm_5_128_layer_call_and_return_conditional_losses_1916624570~FCEDA¢>
7¢4
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ò
P__inference_Batch_Norm_5_128_layer_call_and_return_conditional_losses_1916624604~EFCDA¢>
7¢4
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ª
5__inference_Batch_Norm_5_128_layer_call_fn_1916624537qFCEDA¢>
7¢4
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª
5__inference_Batch_Norm_5_128_layer_call_fn_1916624550qEFCDA¢>
7¢4
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÒ
P__inference_Batch_Norm_5_256_layer_call_and_return_conditional_losses_1916624900~yvxwA¢>
7¢4
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ò
P__inference_Batch_Norm_5_256_layer_call_and_return_conditional_losses_1916624934~xyvwA¢>
7¢4
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ª
5__inference_Batch_Norm_5_256_layer_call_fn_1916624867qyvxwA¢>
7¢4
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª
5__inference_Batch_Norm_5_256_layer_call_fn_1916624880qxyvwA¢>
7¢4
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÒ
P__inference_Batch_Norm_7_128_layer_call_and_return_conditional_losses_1916624650~PMONA¢>
7¢4
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ò
P__inference_Batch_Norm_7_128_layer_call_and_return_conditional_losses_1916624684~OPMNA¢>
7¢4
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ª
5__inference_Batch_Norm_7_128_layer_call_fn_1916624617qPMONA¢>
7¢4
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª
5__inference_Batch_Norm_7_128_layer_call_fn_1916624630qOPMNA¢>
7¢4
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ×
P__inference_Batch_Norm_7_256_layer_call_and_return_conditional_losses_1916624980A¢>
7¢4
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ×
P__inference_Batch_Norm_7_256_layer_call_and_return_conditional_losses_1916625014A¢>
7¢4
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ®
5__inference_Batch_Norm_7_256_layer_call_fn_1916624947uA¢>
7¢4
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ®
5__inference_Batch_Norm_7_256_layer_call_fn_1916624960uA¢>
7¢4
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿµ
S__inference_Concatenate_Dropout_layer_call_and_return_conditional_losses_1916625044^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 µ
S__inference_Concatenate_Dropout_layer_call_and_return_conditional_losses_1916625056^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
8__inference_Concatenate_Dropout_layer_call_fn_1916625034Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
8__inference_Concatenate_Dropout_layer_call_fn_1916625039Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿÈ
L__inference_Conv1D_3_128_layer_call_and_return_conditional_losses_1916624384x#$=¢:
3¢0
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
  
1__inference_Conv1D_3_128_layer_call_fn_1916624363k#$=¢:
3¢0
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
L__inference_Conv1D_3_256_layer_call_and_return_conditional_losses_1916624714xVW=¢:
3¢0
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
  
1__inference_Conv1D_3_256_layer_call_fn_1916624693kVW=¢:
3¢0
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
L__inference_Conv1D_5_128_layer_call_and_return_conditional_losses_1916624414x*+=¢:
3¢0
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
  
1__inference_Conv1D_5_128_layer_call_fn_1916624393k*+=¢:
3¢0
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
L__inference_Conv1D_5_256_layer_call_and_return_conditional_losses_1916624744x]^=¢:
3¢0
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
  
1__inference_Conv1D_5_256_layer_call_fn_1916624723k]^=¢:
3¢0
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
L__inference_Conv1D_7_128_layer_call_and_return_conditional_losses_1916624444x12=¢:
3¢0
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
  
1__inference_Conv1D_7_128_layer_call_fn_1916624423k12=¢:
3¢0
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
L__inference_Conv1D_7_256_layer_call_and_return_conditional_losses_1916624774xde=¢:
3¢0
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
  
1__inference_Conv1D_7_256_layer_call_fn_1916624753kde=¢:
3¢0
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÍ
Q__inference_Embedding_Dropout_layer_call_and_return_conditional_losses_1916624342xA¢>
7¢4
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Í
Q__inference_Embedding_Dropout_layer_call_and_return_conditional_losses_1916624354xA¢>
7¢4
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¥
6__inference_Embedding_Dropout_layer_call_fn_1916624332kA¢>
7¢4
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¥
6__inference_Embedding_Dropout_layer_call_fn_1916624337kA¢>
7¢4
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÅ
J__inference_Max_Pool_3_layer_call_and_return_conditional_losses_1916622503wE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
/__inference_Max_Pool_3_layer_call_fn_1916622509jE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÅ
J__inference_Max_Pool_5_layer_call_and_return_conditional_losses_1916622516wE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
/__inference_Max_Pool_5_layer_call_fn_1916622522jE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÅ
J__inference_Max_Pool_7_layer_call_and_return_conditional_losses_1916622529wE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
/__inference_Max_Pool_7_layer_call_fn_1916622535jE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
P__inference_Pool_Concatenate_layer_call_and_return_conditional_losses_1916625029¬¢~
w¢t
ro
# 
inputs/0ÿÿÿÿÿÿÿÿÿ
# 
inputs/1ÿÿÿÿÿÿÿÿÿ
# 
inputs/2ÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 Ù
5__inference_Pool_Concatenate_layer_call_fn_1916625021¢~
w¢t
ro
# 
inputs/0ÿÿÿÿÿÿÿÿÿ
# 
inputs/1ÿÿÿÿÿÿÿÿÿ
# 
inputs/2ÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ­
J__inference_Prediction_layer_call_and_return_conditional_losses_1916625076_¢£0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
/__inference_Prediction_layer_call_fn_1916625065R¢£0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿí
L__inference_Tree_Encoder_layer_call_and_return_conditional_losses_1916623447,12*+#$PMONFCED<9;:de]^VWyvxwolnm¢£E¢B
;¢8
.+
Inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 í
L__inference_Tree_Encoder_layer_call_and_return_conditional_losses_1916623546,12*+#$OPMNEFCD;<9:de]^VWxyvwnolm¢£E¢B
;¢8
.+
Inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 í
L__inference_Tree_Encoder_layer_call_and_return_conditional_losses_1916624010,12*+#$PMONFCED<9;:de]^VWyvxwolnm¢£E¢B
;¢8
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 í
L__inference_Tree_Encoder_layer_call_and_return_conditional_losses_1916624327,12*+#$OPMNEFCD;<9:de]^VWxyvwnolm¢£E¢B
;¢8
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Å
1__inference_Tree_Encoder_layer_call_fn_1916622882,12*+#$PMONFCED<9;:de]^VWyvxwolnm¢£E¢B
;¢8
.+
Inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿÅ
1__inference_Tree_Encoder_layer_call_fn_1916623348,12*+#$OPMNEFCD;<9:de]^VWxyvwnolm¢£E¢B
;¢8
.+
Inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿÅ
1__inference_Tree_Encoder_layer_call_fn_1916623710,12*+#$PMONFCED<9;:de]^VWyvxwolnm¢£E¢B
;¢8
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿÅ
1__inference_Tree_Encoder_layer_call_fn_1916623791,12*+#$OPMNEFCD;<9:de]^VWxyvwnolm¢£E¢B
;¢8
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿÐ
%__inference__wrapped_model_1916621524¦,12*+#$PMONFCED<9;:de]^VWyvxwolnm¢£=¢:
3¢0
.+
Inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "7ª4
2

Prediction$!

PredictionÿÿÿÿÿÿÿÿÿÝ
(__inference_signature_wrapper_1916623629°,12*+#$PMONFCED<9;:de]^VWyvxwolnm¢£G¢D
¢ 
=ª:
8
Inputs.+
Inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"7ª4
2

Prediction$!

Predictionÿÿÿÿÿÿÿÿÿ