Й▄!
О▐
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
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
Ы
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
М
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
delete_old_dirsbool(И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р
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
dtypetypeИ
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
list(type)(0И
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
╛
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
executor_typestring И
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
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718че
И
Conv1D_3_128/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*$
shared_nameConv1D_3_128/kernel
Б
'Conv1D_3_128/kernel/Read/ReadVariableOpReadVariableOpConv1D_3_128/kernel*$
_output_shapes
:АА*
dtype0
{
Conv1D_3_128/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_nameConv1D_3_128/bias
t
%Conv1D_3_128/bias/Read/ReadVariableOpReadVariableOpConv1D_3_128/bias*
_output_shapes	
:А*
dtype0
И
Conv1D_5_128/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*$
shared_nameConv1D_5_128/kernel
Б
'Conv1D_5_128/kernel/Read/ReadVariableOpReadVariableOpConv1D_5_128/kernel*$
_output_shapes
:АА*
dtype0
{
Conv1D_5_128/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_nameConv1D_5_128/bias
t
%Conv1D_5_128/bias/Read/ReadVariableOpReadVariableOpConv1D_5_128/bias*
_output_shapes	
:А*
dtype0
И
Conv1D_7_128/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*$
shared_nameConv1D_7_128/kernel
Б
'Conv1D_7_128/kernel/Read/ReadVariableOpReadVariableOpConv1D_7_128/kernel*$
_output_shapes
:АА*
dtype0
{
Conv1D_7_128/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_nameConv1D_7_128/bias
t
%Conv1D_7_128/bias/Read/ReadVariableOpReadVariableOpConv1D_7_128/bias*
_output_shapes	
:А*
dtype0
Е
Batch_Norm_3_128/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*'
shared_nameBatch_Norm_3_128/gamma
~
*Batch_Norm_3_128/gamma/Read/ReadVariableOpReadVariableOpBatch_Norm_3_128/gamma*
_output_shapes	
:А*
dtype0
Г
Batch_Norm_3_128/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameBatch_Norm_3_128/beta
|
)Batch_Norm_3_128/beta/Read/ReadVariableOpReadVariableOpBatch_Norm_3_128/beta*
_output_shapes	
:А*
dtype0
С
Batch_Norm_3_128/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*-
shared_nameBatch_Norm_3_128/moving_mean
К
0Batch_Norm_3_128/moving_mean/Read/ReadVariableOpReadVariableOpBatch_Norm_3_128/moving_mean*
_output_shapes	
:А*
dtype0
Щ
 Batch_Norm_3_128/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*1
shared_name" Batch_Norm_3_128/moving_variance
Т
4Batch_Norm_3_128/moving_variance/Read/ReadVariableOpReadVariableOp Batch_Norm_3_128/moving_variance*
_output_shapes	
:А*
dtype0
Е
Batch_Norm_5_128/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*'
shared_nameBatch_Norm_5_128/gamma
~
*Batch_Norm_5_128/gamma/Read/ReadVariableOpReadVariableOpBatch_Norm_5_128/gamma*
_output_shapes	
:А*
dtype0
Г
Batch_Norm_5_128/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameBatch_Norm_5_128/beta
|
)Batch_Norm_5_128/beta/Read/ReadVariableOpReadVariableOpBatch_Norm_5_128/beta*
_output_shapes	
:А*
dtype0
С
Batch_Norm_5_128/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*-
shared_nameBatch_Norm_5_128/moving_mean
К
0Batch_Norm_5_128/moving_mean/Read/ReadVariableOpReadVariableOpBatch_Norm_5_128/moving_mean*
_output_shapes	
:А*
dtype0
Щ
 Batch_Norm_5_128/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*1
shared_name" Batch_Norm_5_128/moving_variance
Т
4Batch_Norm_5_128/moving_variance/Read/ReadVariableOpReadVariableOp Batch_Norm_5_128/moving_variance*
_output_shapes	
:А*
dtype0
Е
Batch_Norm_7_128/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*'
shared_nameBatch_Norm_7_128/gamma
~
*Batch_Norm_7_128/gamma/Read/ReadVariableOpReadVariableOpBatch_Norm_7_128/gamma*
_output_shapes	
:А*
dtype0
Г
Batch_Norm_7_128/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameBatch_Norm_7_128/beta
|
)Batch_Norm_7_128/beta/Read/ReadVariableOpReadVariableOpBatch_Norm_7_128/beta*
_output_shapes	
:А*
dtype0
С
Batch_Norm_7_128/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*-
shared_nameBatch_Norm_7_128/moving_mean
К
0Batch_Norm_7_128/moving_mean/Read/ReadVariableOpReadVariableOpBatch_Norm_7_128/moving_mean*
_output_shapes	
:А*
dtype0
Щ
 Batch_Norm_7_128/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*1
shared_name" Batch_Norm_7_128/moving_variance
Т
4Batch_Norm_7_128/moving_variance/Read/ReadVariableOpReadVariableOp Batch_Norm_7_128/moving_variance*
_output_shapes	
:А*
dtype0
И
Conv1D_3_256/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*$
shared_nameConv1D_3_256/kernel
Б
'Conv1D_3_256/kernel/Read/ReadVariableOpReadVariableOpConv1D_3_256/kernel*$
_output_shapes
:АА*
dtype0
{
Conv1D_3_256/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_nameConv1D_3_256/bias
t
%Conv1D_3_256/bias/Read/ReadVariableOpReadVariableOpConv1D_3_256/bias*
_output_shapes	
:А*
dtype0
И
Conv1D_5_256/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*$
shared_nameConv1D_5_256/kernel
Б
'Conv1D_5_256/kernel/Read/ReadVariableOpReadVariableOpConv1D_5_256/kernel*$
_output_shapes
:АА*
dtype0
{
Conv1D_5_256/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_nameConv1D_5_256/bias
t
%Conv1D_5_256/bias/Read/ReadVariableOpReadVariableOpConv1D_5_256/bias*
_output_shapes	
:А*
dtype0
И
Conv1D_7_256/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*$
shared_nameConv1D_7_256/kernel
Б
'Conv1D_7_256/kernel/Read/ReadVariableOpReadVariableOpConv1D_7_256/kernel*$
_output_shapes
:АА*
dtype0
{
Conv1D_7_256/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_nameConv1D_7_256/bias
t
%Conv1D_7_256/bias/Read/ReadVariableOpReadVariableOpConv1D_7_256/bias*
_output_shapes	
:А*
dtype0
Е
Batch_Norm_3_256/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*'
shared_nameBatch_Norm_3_256/gamma
~
*Batch_Norm_3_256/gamma/Read/ReadVariableOpReadVariableOpBatch_Norm_3_256/gamma*
_output_shapes	
:А*
dtype0
Г
Batch_Norm_3_256/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameBatch_Norm_3_256/beta
|
)Batch_Norm_3_256/beta/Read/ReadVariableOpReadVariableOpBatch_Norm_3_256/beta*
_output_shapes	
:А*
dtype0
С
Batch_Norm_3_256/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*-
shared_nameBatch_Norm_3_256/moving_mean
К
0Batch_Norm_3_256/moving_mean/Read/ReadVariableOpReadVariableOpBatch_Norm_3_256/moving_mean*
_output_shapes	
:А*
dtype0
Щ
 Batch_Norm_3_256/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*1
shared_name" Batch_Norm_3_256/moving_variance
Т
4Batch_Norm_3_256/moving_variance/Read/ReadVariableOpReadVariableOp Batch_Norm_3_256/moving_variance*
_output_shapes	
:А*
dtype0
Е
Batch_Norm_5_256/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*'
shared_nameBatch_Norm_5_256/gamma
~
*Batch_Norm_5_256/gamma/Read/ReadVariableOpReadVariableOpBatch_Norm_5_256/gamma*
_output_shapes	
:А*
dtype0
Г
Batch_Norm_5_256/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameBatch_Norm_5_256/beta
|
)Batch_Norm_5_256/beta/Read/ReadVariableOpReadVariableOpBatch_Norm_5_256/beta*
_output_shapes	
:А*
dtype0
С
Batch_Norm_5_256/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*-
shared_nameBatch_Norm_5_256/moving_mean
К
0Batch_Norm_5_256/moving_mean/Read/ReadVariableOpReadVariableOpBatch_Norm_5_256/moving_mean*
_output_shapes	
:А*
dtype0
Щ
 Batch_Norm_5_256/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*1
shared_name" Batch_Norm_5_256/moving_variance
Т
4Batch_Norm_5_256/moving_variance/Read/ReadVariableOpReadVariableOp Batch_Norm_5_256/moving_variance*
_output_shapes	
:А*
dtype0
Е
Batch_Norm_7_256/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*'
shared_nameBatch_Norm_7_256/gamma
~
*Batch_Norm_7_256/gamma/Read/ReadVariableOpReadVariableOpBatch_Norm_7_256/gamma*
_output_shapes	
:А*
dtype0
Г
Batch_Norm_7_256/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameBatch_Norm_7_256/beta
|
)Batch_Norm_7_256/beta/Read/ReadVariableOpReadVariableOpBatch_Norm_7_256/beta*
_output_shapes	
:А*
dtype0
С
Batch_Norm_7_256/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*-
shared_nameBatch_Norm_7_256/moving_mean
К
0Batch_Norm_7_256/moving_mean/Read/ReadVariableOpReadVariableOpBatch_Norm_7_256/moving_mean*
_output_shapes	
:А*
dtype0
Щ
 Batch_Norm_7_256/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*1
shared_name" Batch_Norm_7_256/moving_variance
Т
4Batch_Norm_7_256/moving_variance/Read/ReadVariableOpReadVariableOp Batch_Norm_7_256/moving_variance*
_output_shapes	
:А*
dtype0

Prediction/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*"
shared_namePrediction/kernel
x
%Prediction/kernel/Read/ReadVariableOpReadVariableOpPrediction/kernel*
_output_shapes
:	А*
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

NoOpNoOp
╪^
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*У^
valueЙ^BЖ^ B ]
┼
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
trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
R
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
 trainable_variables
!regularization_losses
"	variables
#	keras_api
h

$kernel
%bias
&trainable_variables
'regularization_losses
(	variables
)	keras_api
h

*kernel
+bias
,trainable_variables
-regularization_losses
.	variables
/	keras_api
Ч
0axis
	1gamma
2beta
3moving_mean
4moving_variance
5trainable_variables
6regularization_losses
7	variables
8	keras_api
Ч
9axis
	:gamma
;beta
<moving_mean
=moving_variance
>trainable_variables
?regularization_losses
@	variables
A	keras_api
Ч
Baxis
	Cgamma
Dbeta
Emoving_mean
Fmoving_variance
Gtrainable_variables
Hregularization_losses
I	variables
J	keras_api
h

Kkernel
Lbias
Mtrainable_variables
Nregularization_losses
O	variables
P	keras_api
h

Qkernel
Rbias
Strainable_variables
Tregularization_losses
U	variables
V	keras_api
h

Wkernel
Xbias
Ytrainable_variables
Zregularization_losses
[	variables
\	keras_api
Ч
]axis
	^gamma
_beta
`moving_mean
amoving_variance
btrainable_variables
cregularization_losses
d	variables
e	keras_api
Ч
faxis
	ggamma
hbeta
imoving_mean
jmoving_variance
ktrainable_variables
lregularization_losses
m	variables
n	keras_api
Ч
oaxis
	pgamma
qbeta
rmoving_mean
smoving_variance
ttrainable_variables
uregularization_losses
v	variables
w	keras_api
R
xtrainable_variables
yregularization_losses
z	variables
{	keras_api
R
|trainable_variables
}regularization_losses
~	variables
	keras_api
V
Аtrainable_variables
Бregularization_losses
В	variables
Г	keras_api
V
Дtrainable_variables
Еregularization_losses
Ж	variables
З	keras_api
V
Иtrainable_variables
Йregularization_losses
К	variables
Л	keras_api
n
Мkernel
	Нbias
Оtrainable_variables
Пregularization_losses
Р	variables
С	keras_api
╚
0
1
$2
%3
*4
+5
16
27
:8
;9
C10
D11
K12
L13
Q14
R15
W16
X17
^18
_19
g20
h21
p22
q23
М24
Н25
 
и
0
1
$2
%3
*4
+5
16
27
38
49
:10
;11
<12
=13
C14
D15
E16
F17
K18
L19
Q20
R21
W22
X23
^24
_25
`26
a27
g28
h29
i30
j31
p32
q33
r34
s35
М36
Н37
▓
trainable_variables
Тnon_trainable_variables
 Уlayer_regularization_losses
regularization_losses
	variables
Фmetrics
Хlayers
Цlayer_metrics
 
 
 
 
▓
trainable_variables
Чnon_trainable_variables
 Шlayer_regularization_losses
regularization_losses
	variables
Щmetrics
Ъlayers
Ыlayer_metrics
_]
VARIABLE_VALUEConv1D_3_128/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEConv1D_3_128/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
▓
 trainable_variables
Ьnon_trainable_variables
 Эlayer_regularization_losses
!regularization_losses
"	variables
Юmetrics
Яlayers
аlayer_metrics
_]
VARIABLE_VALUEConv1D_5_128/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEConv1D_5_128/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

$0
%1
 

$0
%1
▓
&trainable_variables
бnon_trainable_variables
 вlayer_regularization_losses
'regularization_losses
(	variables
гmetrics
дlayers
еlayer_metrics
_]
VARIABLE_VALUEConv1D_7_128/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEConv1D_7_128/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

*0
+1
 

*0
+1
▓
,trainable_variables
жnon_trainable_variables
 зlayer_regularization_losses
-regularization_losses
.	variables
иmetrics
йlayers
кlayer_metrics
 
a_
VARIABLE_VALUEBatch_Norm_3_128/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEBatch_Norm_3_128/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEBatch_Norm_3_128/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE Batch_Norm_3_128/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

10
21
 

10
21
32
43
▓
5trainable_variables
лnon_trainable_variables
 мlayer_regularization_losses
6regularization_losses
7	variables
нmetrics
оlayers
пlayer_metrics
 
a_
VARIABLE_VALUEBatch_Norm_5_128/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEBatch_Norm_5_128/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEBatch_Norm_5_128/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE Batch_Norm_5_128/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

:0
;1
 

:0
;1
<2
=3
▓
>trainable_variables
░non_trainable_variables
 ▒layer_regularization_losses
?regularization_losses
@	variables
▓metrics
│layers
┤layer_metrics
 
a_
VARIABLE_VALUEBatch_Norm_7_128/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEBatch_Norm_7_128/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEBatch_Norm_7_128/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE Batch_Norm_7_128/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

C0
D1
 

C0
D1
E2
F3
▓
Gtrainable_variables
╡non_trainable_variables
 ╢layer_regularization_losses
Hregularization_losses
I	variables
╖metrics
╕layers
╣layer_metrics
_]
VARIABLE_VALUEConv1D_3_256/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEConv1D_3_256/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

K0
L1
 

K0
L1
▓
Mtrainable_variables
║non_trainable_variables
 ╗layer_regularization_losses
Nregularization_losses
O	variables
╝metrics
╜layers
╛layer_metrics
_]
VARIABLE_VALUEConv1D_5_256/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEConv1D_5_256/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

Q0
R1
 

Q0
R1
▓
Strainable_variables
┐non_trainable_variables
 └layer_regularization_losses
Tregularization_losses
U	variables
┴metrics
┬layers
├layer_metrics
_]
VARIABLE_VALUEConv1D_7_256/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEConv1D_7_256/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

W0
X1
 

W0
X1
▓
Ytrainable_variables
─non_trainable_variables
 ┼layer_regularization_losses
Zregularization_losses
[	variables
╞metrics
╟layers
╚layer_metrics
 
a_
VARIABLE_VALUEBatch_Norm_3_256/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEBatch_Norm_3_256/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEBatch_Norm_3_256/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE Batch_Norm_3_256/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

^0
_1
 

^0
_1
`2
a3
▓
btrainable_variables
╔non_trainable_variables
 ╩layer_regularization_losses
cregularization_losses
d	variables
╦metrics
╠layers
═layer_metrics
 
b`
VARIABLE_VALUEBatch_Norm_5_256/gamma6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEBatch_Norm_5_256/beta5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEBatch_Norm_5_256/moving_mean<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE Batch_Norm_5_256/moving_variance@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

g0
h1
 

g0
h1
i2
j3
▓
ktrainable_variables
╬non_trainable_variables
 ╧layer_regularization_losses
lregularization_losses
m	variables
╨metrics
╤layers
╥layer_metrics
 
b`
VARIABLE_VALUEBatch_Norm_7_256/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEBatch_Norm_7_256/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEBatch_Norm_7_256/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE Batch_Norm_7_256/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

p0
q1
 

p0
q1
r2
s3
▓
ttrainable_variables
╙non_trainable_variables
 ╘layer_regularization_losses
uregularization_losses
v	variables
╒metrics
╓layers
╫layer_metrics
 
 
 
▓
xtrainable_variables
╪non_trainable_variables
 ┘layer_regularization_losses
yregularization_losses
z	variables
┌metrics
█layers
▄layer_metrics
 
 
 
▓
|trainable_variables
▌non_trainable_variables
 ▐layer_regularization_losses
}regularization_losses
~	variables
▀metrics
рlayers
сlayer_metrics
 
 
 
╡
Аtrainable_variables
тnon_trainable_variables
 уlayer_regularization_losses
Бregularization_losses
В	variables
фmetrics
хlayers
цlayer_metrics
 
 
 
╡
Дtrainable_variables
чnon_trainable_variables
 шlayer_regularization_losses
Еregularization_losses
Ж	variables
щmetrics
ъlayers
ыlayer_metrics
 
 
 
╡
Иtrainable_variables
ьnon_trainable_variables
 эlayer_regularization_losses
Йregularization_losses
К	variables
юmetrics
яlayers
Ёlayer_metrics
^\
VARIABLE_VALUEPrediction/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEPrediction/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE

М0
Н1
 

М0
Н1
╡
Оtrainable_variables
ёnon_trainable_variables
 Єlayer_regularization_losses
Пregularization_losses
Р	variables
єmetrics
Їlayers
їlayer_metrics
V
30
41
<2
=3
E4
F5
`6
a7
i8
j9
r10
s11
 
 
Ц
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
30
41
 
 
 
 

<0
=1
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
`0
a1
 
 
 
 

i0
j1
 
 
 
 

r0
s1
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
Х
serving_default_InputsPlaceholder*5
_output_shapes#
!:                  А*
dtype0**
shape!:                  А
▓

StatefulPartitionedCallStatefulPartitionedCallserving_default_InputsConv1D_7_128/kernelConv1D_7_128/biasConv1D_5_128/kernelConv1D_5_128/biasConv1D_3_128/kernelConv1D_3_128/bias Batch_Norm_7_128/moving_varianceBatch_Norm_7_128/gammaBatch_Norm_7_128/moving_meanBatch_Norm_7_128/beta Batch_Norm_5_128/moving_varianceBatch_Norm_5_128/gammaBatch_Norm_5_128/moving_meanBatch_Norm_5_128/beta Batch_Norm_3_128/moving_varianceBatch_Norm_3_128/gammaBatch_Norm_3_128/moving_meanBatch_Norm_3_128/betaConv1D_7_256/kernelConv1D_7_256/biasConv1D_5_256/kernelConv1D_5_256/biasConv1D_3_256/kernelConv1D_3_256/bias Batch_Norm_7_256/moving_varianceBatch_Norm_7_256/gammaBatch_Norm_7_256/moving_meanBatch_Norm_7_256/beta Batch_Norm_5_256/moving_varianceBatch_Norm_5_256/gammaBatch_Norm_5_256/moving_meanBatch_Norm_5_256/beta Batch_Norm_3_256/moving_varianceBatch_Norm_3_256/gammaBatch_Norm_3_256/moving_meanBatch_Norm_3_256/betaPrediction/kernelPrediction/bias*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8В *1
f,R*
(__inference_signature_wrapper_1294349940
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ю
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename'Conv1D_3_128/kernel/Read/ReadVariableOp%Conv1D_3_128/bias/Read/ReadVariableOp'Conv1D_5_128/kernel/Read/ReadVariableOp%Conv1D_5_128/bias/Read/ReadVariableOp'Conv1D_7_128/kernel/Read/ReadVariableOp%Conv1D_7_128/bias/Read/ReadVariableOp*Batch_Norm_3_128/gamma/Read/ReadVariableOp)Batch_Norm_3_128/beta/Read/ReadVariableOp0Batch_Norm_3_128/moving_mean/Read/ReadVariableOp4Batch_Norm_3_128/moving_variance/Read/ReadVariableOp*Batch_Norm_5_128/gamma/Read/ReadVariableOp)Batch_Norm_5_128/beta/Read/ReadVariableOp0Batch_Norm_5_128/moving_mean/Read/ReadVariableOp4Batch_Norm_5_128/moving_variance/Read/ReadVariableOp*Batch_Norm_7_128/gamma/Read/ReadVariableOp)Batch_Norm_7_128/beta/Read/ReadVariableOp0Batch_Norm_7_128/moving_mean/Read/ReadVariableOp4Batch_Norm_7_128/moving_variance/Read/ReadVariableOp'Conv1D_3_256/kernel/Read/ReadVariableOp%Conv1D_3_256/bias/Read/ReadVariableOp'Conv1D_5_256/kernel/Read/ReadVariableOp%Conv1D_5_256/bias/Read/ReadVariableOp'Conv1D_7_256/kernel/Read/ReadVariableOp%Conv1D_7_256/bias/Read/ReadVariableOp*Batch_Norm_3_256/gamma/Read/ReadVariableOp)Batch_Norm_3_256/beta/Read/ReadVariableOp0Batch_Norm_3_256/moving_mean/Read/ReadVariableOp4Batch_Norm_3_256/moving_variance/Read/ReadVariableOp*Batch_Norm_5_256/gamma/Read/ReadVariableOp)Batch_Norm_5_256/beta/Read/ReadVariableOp0Batch_Norm_5_256/moving_mean/Read/ReadVariableOp4Batch_Norm_5_256/moving_variance/Read/ReadVariableOp*Batch_Norm_7_256/gamma/Read/ReadVariableOp)Batch_Norm_7_256/beta/Read/ReadVariableOp0Batch_Norm_7_256/moving_mean/Read/ReadVariableOp4Batch_Norm_7_256/moving_variance/Read/ReadVariableOp%Prediction/kernel/Read/ReadVariableOp#Prediction/bias/Read/ReadVariableOpConst*3
Tin,
*2(*
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
GPU2*0J 8В *,
f'R%
#__inference__traced_save_1294351524
ё	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameConv1D_3_128/kernelConv1D_3_128/biasConv1D_5_128/kernelConv1D_5_128/biasConv1D_7_128/kernelConv1D_7_128/biasBatch_Norm_3_128/gammaBatch_Norm_3_128/betaBatch_Norm_3_128/moving_mean Batch_Norm_3_128/moving_varianceBatch_Norm_5_128/gammaBatch_Norm_5_128/betaBatch_Norm_5_128/moving_mean Batch_Norm_5_128/moving_varianceBatch_Norm_7_128/gammaBatch_Norm_7_128/betaBatch_Norm_7_128/moving_mean Batch_Norm_7_128/moving_varianceConv1D_3_256/kernelConv1D_3_256/biasConv1D_5_256/kernelConv1D_5_256/biasConv1D_7_256/kernelConv1D_7_256/biasBatch_Norm_3_256/gammaBatch_Norm_3_256/betaBatch_Norm_3_256/moving_mean Batch_Norm_3_256/moving_varianceBatch_Norm_5_256/gammaBatch_Norm_5_256/betaBatch_Norm_5_256/moving_mean Batch_Norm_5_256/moving_varianceBatch_Norm_7_256/gammaBatch_Norm_7_256/betaBatch_Norm_7_256/moving_mean Batch_Norm_7_256/moving_variancePrediction/kernelPrediction/bias*2
Tin+
)2'*
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
GPU2*0J 8В */
f*R(
&__inference__traced_restore_1294351648╠╨
ё
│
P__inference_Batch_Norm_7_128_layer_call_and_return_conditional_losses_1294348183

inputs0
!batchnorm_readvariableop_resource:	А4
%batchnorm_mul_readvariableop_resource:	А2
#batchnorm_readvariableop_1_resource:	А2
#batchnorm_readvariableop_2_resource:	А
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/add_1щ
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  А: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
╝
╘
5__inference_Batch_Norm_3_128_layer_call_fn_1294350781

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_Batch_Norm_3_128_layer_call_and_return_conditional_losses_12943479192
StatefulPartitionedCallЬ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
ё
│
P__inference_Batch_Norm_5_128_layer_call_and_return_conditional_losses_1294350881

inputs0
!batchnorm_readvariableop_resource:	А4
%batchnorm_mul_readvariableop_resource:	А2
#batchnorm_readvariableop_1_resource:	А2
#batchnorm_readvariableop_2_resource:	А
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/add_1щ
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  А: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
ё
│
P__inference_Batch_Norm_7_128_layer_call_and_return_conditional_losses_1294350961

inputs0
!batchnorm_readvariableop_resource:	А4
%batchnorm_mul_readvariableop_resource:	А2
#batchnorm_readvariableop_1_resource:	А2
#batchnorm_readvariableop_2_resource:	А
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/add_1щ
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  А: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
№*
э
P__inference_Batch_Norm_7_256_layer_call_and_return_conditional_losses_1294348729

inputs6
'assignmovingavg_readvariableop_resource:	А8
)assignmovingavg_1_readvariableop_resource:	А4
%batchnorm_mul_readvariableop_resource:	А0
!batchnorm_readvariableop_resource:	А
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesФ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/meanБ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:А2
moments/StopGradient▓
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:                  А2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╖
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/varianceВ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/SqueezeК
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayе
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOpЩ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:А2
AssignMovingAvg/subР
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:А2
AssignMovingAvg/mul┐
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
╫#<2
AssignMovingAvg_1/decayл
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOpб
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:А2
AssignMovingAvg_1/subШ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:А2
AssignMovingAvg_1/mul╔
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
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/add_1Щ
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  А: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
г╚
▄"
L__inference_Tree_Encoder_layer_call_and_return_conditional_losses_1294350321

inputsP
8conv1d_7_128_conv1d_expanddims_1_readvariableop_resource:АА;
,conv1d_7_128_biasadd_readvariableop_resource:	АP
8conv1d_5_128_conv1d_expanddims_1_readvariableop_resource:АА;
,conv1d_5_128_biasadd_readvariableop_resource:	АP
8conv1d_3_128_conv1d_expanddims_1_readvariableop_resource:АА;
,conv1d_3_128_biasadd_readvariableop_resource:	АA
2batch_norm_7_128_batchnorm_readvariableop_resource:	АE
6batch_norm_7_128_batchnorm_mul_readvariableop_resource:	АC
4batch_norm_7_128_batchnorm_readvariableop_1_resource:	АC
4batch_norm_7_128_batchnorm_readvariableop_2_resource:	АA
2batch_norm_5_128_batchnorm_readvariableop_resource:	АE
6batch_norm_5_128_batchnorm_mul_readvariableop_resource:	АC
4batch_norm_5_128_batchnorm_readvariableop_1_resource:	АC
4batch_norm_5_128_batchnorm_readvariableop_2_resource:	АA
2batch_norm_3_128_batchnorm_readvariableop_resource:	АE
6batch_norm_3_128_batchnorm_mul_readvariableop_resource:	АC
4batch_norm_3_128_batchnorm_readvariableop_1_resource:	АC
4batch_norm_3_128_batchnorm_readvariableop_2_resource:	АP
8conv1d_7_256_conv1d_expanddims_1_readvariableop_resource:АА;
,conv1d_7_256_biasadd_readvariableop_resource:	АP
8conv1d_5_256_conv1d_expanddims_1_readvariableop_resource:АА;
,conv1d_5_256_biasadd_readvariableop_resource:	АP
8conv1d_3_256_conv1d_expanddims_1_readvariableop_resource:АА;
,conv1d_3_256_biasadd_readvariableop_resource:	АA
2batch_norm_7_256_batchnorm_readvariableop_resource:	АE
6batch_norm_7_256_batchnorm_mul_readvariableop_resource:	АC
4batch_norm_7_256_batchnorm_readvariableop_1_resource:	АC
4batch_norm_7_256_batchnorm_readvariableop_2_resource:	АA
2batch_norm_5_256_batchnorm_readvariableop_resource:	АE
6batch_norm_5_256_batchnorm_mul_readvariableop_resource:	АC
4batch_norm_5_256_batchnorm_readvariableop_1_resource:	АC
4batch_norm_5_256_batchnorm_readvariableop_2_resource:	АA
2batch_norm_3_256_batchnorm_readvariableop_resource:	АE
6batch_norm_3_256_batchnorm_mul_readvariableop_resource:	АC
4batch_norm_3_256_batchnorm_readvariableop_1_resource:	АC
4batch_norm_3_256_batchnorm_readvariableop_2_resource:	А<
)prediction_matmul_readvariableop_resource:	А8
*prediction_biasadd_readvariableop_resource:
identityИв)Batch_Norm_3_128/batchnorm/ReadVariableOpв+Batch_Norm_3_128/batchnorm/ReadVariableOp_1в+Batch_Norm_3_128/batchnorm/ReadVariableOp_2в-Batch_Norm_3_128/batchnorm/mul/ReadVariableOpв)Batch_Norm_3_256/batchnorm/ReadVariableOpв+Batch_Norm_3_256/batchnorm/ReadVariableOp_1в+Batch_Norm_3_256/batchnorm/ReadVariableOp_2в-Batch_Norm_3_256/batchnorm/mul/ReadVariableOpв)Batch_Norm_5_128/batchnorm/ReadVariableOpв+Batch_Norm_5_128/batchnorm/ReadVariableOp_1в+Batch_Norm_5_128/batchnorm/ReadVariableOp_2в-Batch_Norm_5_128/batchnorm/mul/ReadVariableOpв)Batch_Norm_5_256/batchnorm/ReadVariableOpв+Batch_Norm_5_256/batchnorm/ReadVariableOp_1в+Batch_Norm_5_256/batchnorm/ReadVariableOp_2в-Batch_Norm_5_256/batchnorm/mul/ReadVariableOpв)Batch_Norm_7_128/batchnorm/ReadVariableOpв+Batch_Norm_7_128/batchnorm/ReadVariableOp_1в+Batch_Norm_7_128/batchnorm/ReadVariableOp_2в-Batch_Norm_7_128/batchnorm/mul/ReadVariableOpв)Batch_Norm_7_256/batchnorm/ReadVariableOpв+Batch_Norm_7_256/batchnorm/ReadVariableOp_1в+Batch_Norm_7_256/batchnorm/ReadVariableOp_2в-Batch_Norm_7_256/batchnorm/mul/ReadVariableOpв#Conv1D_3_128/BiasAdd/ReadVariableOpв/Conv1D_3_128/conv1d/ExpandDims_1/ReadVariableOpв#Conv1D_3_256/BiasAdd/ReadVariableOpв/Conv1D_3_256/conv1d/ExpandDims_1/ReadVariableOpв#Conv1D_5_128/BiasAdd/ReadVariableOpв/Conv1D_5_128/conv1d/ExpandDims_1/ReadVariableOpв#Conv1D_5_256/BiasAdd/ReadVariableOpв/Conv1D_5_256/conv1d/ExpandDims_1/ReadVariableOpв#Conv1D_7_128/BiasAdd/ReadVariableOpв/Conv1D_7_128/conv1d/ExpandDims_1/ReadVariableOpв#Conv1D_7_256/BiasAdd/ReadVariableOpв/Conv1D_7_256/conv1d/ExpandDims_1/ReadVariableOpв!Prediction/BiasAdd/ReadVariableOpв Prediction/MatMul/ReadVariableOpМ
Embedding_Dropout/IdentityIdentityinputs*
T0*5
_output_shapes#
!:                  А2
Embedding_Dropout/IdentityУ
"Conv1D_7_128/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2$
"Conv1D_7_128/conv1d/ExpandDims/dimф
Conv1D_7_128/conv1d/ExpandDims
ExpandDims#Embedding_Dropout/Identity:output:0+Conv1D_7_128/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#                  А2 
Conv1D_7_128/conv1d/ExpandDimsс
/Conv1D_7_128/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp8conv1d_7_128_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:АА*
dtype021
/Conv1D_7_128/conv1d/ExpandDims_1/ReadVariableOpО
$Conv1D_7_128/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Conv1D_7_128/conv1d/ExpandDims_1/dimэ
 Conv1D_7_128/conv1d/ExpandDims_1
ExpandDims7Conv1D_7_128/conv1d/ExpandDims_1/ReadVariableOp:value:0-Conv1D_7_128/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:АА2"
 Conv1D_7_128/conv1d/ExpandDims_1Ї
Conv1D_7_128/conv1dConv2D'Conv1D_7_128/conv1d/ExpandDims:output:0)Conv1D_7_128/conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#                  А*
paddingSAME*
strides
2
Conv1D_7_128/conv1d├
Conv1D_7_128/conv1d/SqueezeSqueezeConv1D_7_128/conv1d:output:0*
T0*5
_output_shapes#
!:                  А*
squeeze_dims

¤        2
Conv1D_7_128/conv1d/Squeeze┤
#Conv1D_7_128/BiasAdd/ReadVariableOpReadVariableOp,conv1d_7_128_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#Conv1D_7_128/BiasAdd/ReadVariableOp╩
Conv1D_7_128/BiasAddBiasAdd$Conv1D_7_128/conv1d/Squeeze:output:0+Conv1D_7_128/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:                  А2
Conv1D_7_128/BiasAddЦ
Conv1D_7_128/SigmoidSigmoidConv1D_7_128/BiasAdd:output:0*
T0*5
_output_shapes#
!:                  А2
Conv1D_7_128/Sigmoidд
Conv1D_7_128/mulMulConv1D_7_128/BiasAdd:output:0Conv1D_7_128/Sigmoid:y:0*
T0*5
_output_shapes#
!:                  А2
Conv1D_7_128/mulР
Conv1D_7_128/IdentityIdentityConv1D_7_128/mul:z:0*
T0*5
_output_shapes#
!:                  А2
Conv1D_7_128/IdentityЙ
Conv1D_7_128/IdentityN	IdentityNConv1D_7_128/mul:z:0Conv1D_7_128/BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1294350117*V
_output_shapesD
B:                  А:                  А2
Conv1D_7_128/IdentityNУ
"Conv1D_5_128/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2$
"Conv1D_5_128/conv1d/ExpandDims/dimф
Conv1D_5_128/conv1d/ExpandDims
ExpandDims#Embedding_Dropout/Identity:output:0+Conv1D_5_128/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#                  А2 
Conv1D_5_128/conv1d/ExpandDimsс
/Conv1D_5_128/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp8conv1d_5_128_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:АА*
dtype021
/Conv1D_5_128/conv1d/ExpandDims_1/ReadVariableOpО
$Conv1D_5_128/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Conv1D_5_128/conv1d/ExpandDims_1/dimэ
 Conv1D_5_128/conv1d/ExpandDims_1
ExpandDims7Conv1D_5_128/conv1d/ExpandDims_1/ReadVariableOp:value:0-Conv1D_5_128/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:АА2"
 Conv1D_5_128/conv1d/ExpandDims_1Ї
Conv1D_5_128/conv1dConv2D'Conv1D_5_128/conv1d/ExpandDims:output:0)Conv1D_5_128/conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#                  А*
paddingSAME*
strides
2
Conv1D_5_128/conv1d├
Conv1D_5_128/conv1d/SqueezeSqueezeConv1D_5_128/conv1d:output:0*
T0*5
_output_shapes#
!:                  А*
squeeze_dims

¤        2
Conv1D_5_128/conv1d/Squeeze┤
#Conv1D_5_128/BiasAdd/ReadVariableOpReadVariableOp,conv1d_5_128_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#Conv1D_5_128/BiasAdd/ReadVariableOp╩
Conv1D_5_128/BiasAddBiasAdd$Conv1D_5_128/conv1d/Squeeze:output:0+Conv1D_5_128/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:                  А2
Conv1D_5_128/BiasAddЦ
Conv1D_5_128/SigmoidSigmoidConv1D_5_128/BiasAdd:output:0*
T0*5
_output_shapes#
!:                  А2
Conv1D_5_128/Sigmoidд
Conv1D_5_128/mulMulConv1D_5_128/BiasAdd:output:0Conv1D_5_128/Sigmoid:y:0*
T0*5
_output_shapes#
!:                  А2
Conv1D_5_128/mulР
Conv1D_5_128/IdentityIdentityConv1D_5_128/mul:z:0*
T0*5
_output_shapes#
!:                  А2
Conv1D_5_128/IdentityЙ
Conv1D_5_128/IdentityN	IdentityNConv1D_5_128/mul:z:0Conv1D_5_128/BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1294350134*V
_output_shapesD
B:                  А:                  А2
Conv1D_5_128/IdentityNУ
"Conv1D_3_128/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2$
"Conv1D_3_128/conv1d/ExpandDims/dimф
Conv1D_3_128/conv1d/ExpandDims
ExpandDims#Embedding_Dropout/Identity:output:0+Conv1D_3_128/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#                  А2 
Conv1D_3_128/conv1d/ExpandDimsс
/Conv1D_3_128/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp8conv1d_3_128_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:АА*
dtype021
/Conv1D_3_128/conv1d/ExpandDims_1/ReadVariableOpО
$Conv1D_3_128/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Conv1D_3_128/conv1d/ExpandDims_1/dimэ
 Conv1D_3_128/conv1d/ExpandDims_1
ExpandDims7Conv1D_3_128/conv1d/ExpandDims_1/ReadVariableOp:value:0-Conv1D_3_128/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:АА2"
 Conv1D_3_128/conv1d/ExpandDims_1Ї
Conv1D_3_128/conv1dConv2D'Conv1D_3_128/conv1d/ExpandDims:output:0)Conv1D_3_128/conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#                  А*
paddingSAME*
strides
2
Conv1D_3_128/conv1d├
Conv1D_3_128/conv1d/SqueezeSqueezeConv1D_3_128/conv1d:output:0*
T0*5
_output_shapes#
!:                  А*
squeeze_dims

¤        2
Conv1D_3_128/conv1d/Squeeze┤
#Conv1D_3_128/BiasAdd/ReadVariableOpReadVariableOp,conv1d_3_128_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#Conv1D_3_128/BiasAdd/ReadVariableOp╩
Conv1D_3_128/BiasAddBiasAdd$Conv1D_3_128/conv1d/Squeeze:output:0+Conv1D_3_128/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:                  А2
Conv1D_3_128/BiasAddЦ
Conv1D_3_128/SigmoidSigmoidConv1D_3_128/BiasAdd:output:0*
T0*5
_output_shapes#
!:                  А2
Conv1D_3_128/Sigmoidд
Conv1D_3_128/mulMulConv1D_3_128/BiasAdd:output:0Conv1D_3_128/Sigmoid:y:0*
T0*5
_output_shapes#
!:                  А2
Conv1D_3_128/mulР
Conv1D_3_128/IdentityIdentityConv1D_3_128/mul:z:0*
T0*5
_output_shapes#
!:                  А2
Conv1D_3_128/IdentityЙ
Conv1D_3_128/IdentityN	IdentityNConv1D_3_128/mul:z:0Conv1D_3_128/BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1294350151*V
_output_shapesD
B:                  А:                  А2
Conv1D_3_128/IdentityN╞
)Batch_Norm_7_128/batchnorm/ReadVariableOpReadVariableOp2batch_norm_7_128_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)Batch_Norm_7_128/batchnorm/ReadVariableOpЙ
 Batch_Norm_7_128/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2"
 Batch_Norm_7_128/batchnorm/add/y═
Batch_Norm_7_128/batchnorm/addAddV21Batch_Norm_7_128/batchnorm/ReadVariableOp:value:0)Batch_Norm_7_128/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2 
Batch_Norm_7_128/batchnorm/addЧ
 Batch_Norm_7_128/batchnorm/RsqrtRsqrt"Batch_Norm_7_128/batchnorm/add:z:0*
T0*
_output_shapes	
:А2"
 Batch_Norm_7_128/batchnorm/Rsqrt╥
-Batch_Norm_7_128/batchnorm/mul/ReadVariableOpReadVariableOp6batch_norm_7_128_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-Batch_Norm_7_128/batchnorm/mul/ReadVariableOp╩
Batch_Norm_7_128/batchnorm/mulMul$Batch_Norm_7_128/batchnorm/Rsqrt:y:05Batch_Norm_7_128/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2 
Batch_Norm_7_128/batchnorm/mul╨
 Batch_Norm_7_128/batchnorm/mul_1MulConv1D_7_128/IdentityN:output:0"Batch_Norm_7_128/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2"
 Batch_Norm_7_128/batchnorm/mul_1╠
+Batch_Norm_7_128/batchnorm/ReadVariableOp_1ReadVariableOp4batch_norm_7_128_batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02-
+Batch_Norm_7_128/batchnorm/ReadVariableOp_1╩
 Batch_Norm_7_128/batchnorm/mul_2Mul3Batch_Norm_7_128/batchnorm/ReadVariableOp_1:value:0"Batch_Norm_7_128/batchnorm/mul:z:0*
T0*
_output_shapes	
:А2"
 Batch_Norm_7_128/batchnorm/mul_2╠
+Batch_Norm_7_128/batchnorm/ReadVariableOp_2ReadVariableOp4batch_norm_7_128_batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02-
+Batch_Norm_7_128/batchnorm/ReadVariableOp_2╚
Batch_Norm_7_128/batchnorm/subSub3Batch_Norm_7_128/batchnorm/ReadVariableOp_2:value:0$Batch_Norm_7_128/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2 
Batch_Norm_7_128/batchnorm/sub╫
 Batch_Norm_7_128/batchnorm/add_1AddV2$Batch_Norm_7_128/batchnorm/mul_1:z:0"Batch_Norm_7_128/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2"
 Batch_Norm_7_128/batchnorm/add_1╞
)Batch_Norm_5_128/batchnorm/ReadVariableOpReadVariableOp2batch_norm_5_128_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)Batch_Norm_5_128/batchnorm/ReadVariableOpЙ
 Batch_Norm_5_128/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2"
 Batch_Norm_5_128/batchnorm/add/y═
Batch_Norm_5_128/batchnorm/addAddV21Batch_Norm_5_128/batchnorm/ReadVariableOp:value:0)Batch_Norm_5_128/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2 
Batch_Norm_5_128/batchnorm/addЧ
 Batch_Norm_5_128/batchnorm/RsqrtRsqrt"Batch_Norm_5_128/batchnorm/add:z:0*
T0*
_output_shapes	
:А2"
 Batch_Norm_5_128/batchnorm/Rsqrt╥
-Batch_Norm_5_128/batchnorm/mul/ReadVariableOpReadVariableOp6batch_norm_5_128_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-Batch_Norm_5_128/batchnorm/mul/ReadVariableOp╩
Batch_Norm_5_128/batchnorm/mulMul$Batch_Norm_5_128/batchnorm/Rsqrt:y:05Batch_Norm_5_128/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2 
Batch_Norm_5_128/batchnorm/mul╨
 Batch_Norm_5_128/batchnorm/mul_1MulConv1D_5_128/IdentityN:output:0"Batch_Norm_5_128/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2"
 Batch_Norm_5_128/batchnorm/mul_1╠
+Batch_Norm_5_128/batchnorm/ReadVariableOp_1ReadVariableOp4batch_norm_5_128_batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02-
+Batch_Norm_5_128/batchnorm/ReadVariableOp_1╩
 Batch_Norm_5_128/batchnorm/mul_2Mul3Batch_Norm_5_128/batchnorm/ReadVariableOp_1:value:0"Batch_Norm_5_128/batchnorm/mul:z:0*
T0*
_output_shapes	
:А2"
 Batch_Norm_5_128/batchnorm/mul_2╠
+Batch_Norm_5_128/batchnorm/ReadVariableOp_2ReadVariableOp4batch_norm_5_128_batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02-
+Batch_Norm_5_128/batchnorm/ReadVariableOp_2╚
Batch_Norm_5_128/batchnorm/subSub3Batch_Norm_5_128/batchnorm/ReadVariableOp_2:value:0$Batch_Norm_5_128/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2 
Batch_Norm_5_128/batchnorm/sub╫
 Batch_Norm_5_128/batchnorm/add_1AddV2$Batch_Norm_5_128/batchnorm/mul_1:z:0"Batch_Norm_5_128/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2"
 Batch_Norm_5_128/batchnorm/add_1╞
)Batch_Norm_3_128/batchnorm/ReadVariableOpReadVariableOp2batch_norm_3_128_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)Batch_Norm_3_128/batchnorm/ReadVariableOpЙ
 Batch_Norm_3_128/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2"
 Batch_Norm_3_128/batchnorm/add/y═
Batch_Norm_3_128/batchnorm/addAddV21Batch_Norm_3_128/batchnorm/ReadVariableOp:value:0)Batch_Norm_3_128/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2 
Batch_Norm_3_128/batchnorm/addЧ
 Batch_Norm_3_128/batchnorm/RsqrtRsqrt"Batch_Norm_3_128/batchnorm/add:z:0*
T0*
_output_shapes	
:А2"
 Batch_Norm_3_128/batchnorm/Rsqrt╥
-Batch_Norm_3_128/batchnorm/mul/ReadVariableOpReadVariableOp6batch_norm_3_128_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-Batch_Norm_3_128/batchnorm/mul/ReadVariableOp╩
Batch_Norm_3_128/batchnorm/mulMul$Batch_Norm_3_128/batchnorm/Rsqrt:y:05Batch_Norm_3_128/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2 
Batch_Norm_3_128/batchnorm/mul╨
 Batch_Norm_3_128/batchnorm/mul_1MulConv1D_3_128/IdentityN:output:0"Batch_Norm_3_128/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2"
 Batch_Norm_3_128/batchnorm/mul_1╠
+Batch_Norm_3_128/batchnorm/ReadVariableOp_1ReadVariableOp4batch_norm_3_128_batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02-
+Batch_Norm_3_128/batchnorm/ReadVariableOp_1╩
 Batch_Norm_3_128/batchnorm/mul_2Mul3Batch_Norm_3_128/batchnorm/ReadVariableOp_1:value:0"Batch_Norm_3_128/batchnorm/mul:z:0*
T0*
_output_shapes	
:А2"
 Batch_Norm_3_128/batchnorm/mul_2╠
+Batch_Norm_3_128/batchnorm/ReadVariableOp_2ReadVariableOp4batch_norm_3_128_batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02-
+Batch_Norm_3_128/batchnorm/ReadVariableOp_2╚
Batch_Norm_3_128/batchnorm/subSub3Batch_Norm_3_128/batchnorm/ReadVariableOp_2:value:0$Batch_Norm_3_128/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2 
Batch_Norm_3_128/batchnorm/sub╫
 Batch_Norm_3_128/batchnorm/add_1AddV2$Batch_Norm_3_128/batchnorm/mul_1:z:0"Batch_Norm_3_128/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2"
 Batch_Norm_3_128/batchnorm/add_1У
"Conv1D_7_256/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2$
"Conv1D_7_256/conv1d/ExpandDims/dimх
Conv1D_7_256/conv1d/ExpandDims
ExpandDims$Batch_Norm_7_128/batchnorm/add_1:z:0+Conv1D_7_256/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#                  А2 
Conv1D_7_256/conv1d/ExpandDimsс
/Conv1D_7_256/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp8conv1d_7_256_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:АА*
dtype021
/Conv1D_7_256/conv1d/ExpandDims_1/ReadVariableOpО
$Conv1D_7_256/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Conv1D_7_256/conv1d/ExpandDims_1/dimэ
 Conv1D_7_256/conv1d/ExpandDims_1
ExpandDims7Conv1D_7_256/conv1d/ExpandDims_1/ReadVariableOp:value:0-Conv1D_7_256/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:АА2"
 Conv1D_7_256/conv1d/ExpandDims_1Ї
Conv1D_7_256/conv1dConv2D'Conv1D_7_256/conv1d/ExpandDims:output:0)Conv1D_7_256/conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#                  А*
paddingSAME*
strides
2
Conv1D_7_256/conv1d├
Conv1D_7_256/conv1d/SqueezeSqueezeConv1D_7_256/conv1d:output:0*
T0*5
_output_shapes#
!:                  А*
squeeze_dims

¤        2
Conv1D_7_256/conv1d/Squeeze┤
#Conv1D_7_256/BiasAdd/ReadVariableOpReadVariableOp,conv1d_7_256_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#Conv1D_7_256/BiasAdd/ReadVariableOp╩
Conv1D_7_256/BiasAddBiasAdd$Conv1D_7_256/conv1d/Squeeze:output:0+Conv1D_7_256/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:                  А2
Conv1D_7_256/BiasAddЦ
Conv1D_7_256/SigmoidSigmoidConv1D_7_256/BiasAdd:output:0*
T0*5
_output_shapes#
!:                  А2
Conv1D_7_256/Sigmoidд
Conv1D_7_256/mulMulConv1D_7_256/BiasAdd:output:0Conv1D_7_256/Sigmoid:y:0*
T0*5
_output_shapes#
!:                  А2
Conv1D_7_256/mulР
Conv1D_7_256/IdentityIdentityConv1D_7_256/mul:z:0*
T0*5
_output_shapes#
!:                  А2
Conv1D_7_256/IdentityЙ
Conv1D_7_256/IdentityN	IdentityNConv1D_7_256/mul:z:0Conv1D_7_256/BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1294350216*V
_output_shapesD
B:                  А:                  А2
Conv1D_7_256/IdentityNУ
"Conv1D_5_256/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2$
"Conv1D_5_256/conv1d/ExpandDims/dimх
Conv1D_5_256/conv1d/ExpandDims
ExpandDims$Batch_Norm_5_128/batchnorm/add_1:z:0+Conv1D_5_256/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#                  А2 
Conv1D_5_256/conv1d/ExpandDimsс
/Conv1D_5_256/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp8conv1d_5_256_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:АА*
dtype021
/Conv1D_5_256/conv1d/ExpandDims_1/ReadVariableOpО
$Conv1D_5_256/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Conv1D_5_256/conv1d/ExpandDims_1/dimэ
 Conv1D_5_256/conv1d/ExpandDims_1
ExpandDims7Conv1D_5_256/conv1d/ExpandDims_1/ReadVariableOp:value:0-Conv1D_5_256/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:АА2"
 Conv1D_5_256/conv1d/ExpandDims_1Ї
Conv1D_5_256/conv1dConv2D'Conv1D_5_256/conv1d/ExpandDims:output:0)Conv1D_5_256/conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#                  А*
paddingSAME*
strides
2
Conv1D_5_256/conv1d├
Conv1D_5_256/conv1d/SqueezeSqueezeConv1D_5_256/conv1d:output:0*
T0*5
_output_shapes#
!:                  А*
squeeze_dims

¤        2
Conv1D_5_256/conv1d/Squeeze┤
#Conv1D_5_256/BiasAdd/ReadVariableOpReadVariableOp,conv1d_5_256_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#Conv1D_5_256/BiasAdd/ReadVariableOp╩
Conv1D_5_256/BiasAddBiasAdd$Conv1D_5_256/conv1d/Squeeze:output:0+Conv1D_5_256/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:                  А2
Conv1D_5_256/BiasAddЦ
Conv1D_5_256/SigmoidSigmoidConv1D_5_256/BiasAdd:output:0*
T0*5
_output_shapes#
!:                  А2
Conv1D_5_256/Sigmoidд
Conv1D_5_256/mulMulConv1D_5_256/BiasAdd:output:0Conv1D_5_256/Sigmoid:y:0*
T0*5
_output_shapes#
!:                  А2
Conv1D_5_256/mulР
Conv1D_5_256/IdentityIdentityConv1D_5_256/mul:z:0*
T0*5
_output_shapes#
!:                  А2
Conv1D_5_256/IdentityЙ
Conv1D_5_256/IdentityN	IdentityNConv1D_5_256/mul:z:0Conv1D_5_256/BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1294350233*V
_output_shapesD
B:                  А:                  А2
Conv1D_5_256/IdentityNУ
"Conv1D_3_256/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2$
"Conv1D_3_256/conv1d/ExpandDims/dimх
Conv1D_3_256/conv1d/ExpandDims
ExpandDims$Batch_Norm_3_128/batchnorm/add_1:z:0+Conv1D_3_256/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#                  А2 
Conv1D_3_256/conv1d/ExpandDimsс
/Conv1D_3_256/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp8conv1d_3_256_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:АА*
dtype021
/Conv1D_3_256/conv1d/ExpandDims_1/ReadVariableOpО
$Conv1D_3_256/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Conv1D_3_256/conv1d/ExpandDims_1/dimэ
 Conv1D_3_256/conv1d/ExpandDims_1
ExpandDims7Conv1D_3_256/conv1d/ExpandDims_1/ReadVariableOp:value:0-Conv1D_3_256/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:АА2"
 Conv1D_3_256/conv1d/ExpandDims_1Ї
Conv1D_3_256/conv1dConv2D'Conv1D_3_256/conv1d/ExpandDims:output:0)Conv1D_3_256/conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#                  А*
paddingSAME*
strides
2
Conv1D_3_256/conv1d├
Conv1D_3_256/conv1d/SqueezeSqueezeConv1D_3_256/conv1d:output:0*
T0*5
_output_shapes#
!:                  А*
squeeze_dims

¤        2
Conv1D_3_256/conv1d/Squeeze┤
#Conv1D_3_256/BiasAdd/ReadVariableOpReadVariableOp,conv1d_3_256_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#Conv1D_3_256/BiasAdd/ReadVariableOp╩
Conv1D_3_256/BiasAddBiasAdd$Conv1D_3_256/conv1d/Squeeze:output:0+Conv1D_3_256/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:                  А2
Conv1D_3_256/BiasAddЦ
Conv1D_3_256/SigmoidSigmoidConv1D_3_256/BiasAdd:output:0*
T0*5
_output_shapes#
!:                  А2
Conv1D_3_256/Sigmoidд
Conv1D_3_256/mulMulConv1D_3_256/BiasAdd:output:0Conv1D_3_256/Sigmoid:y:0*
T0*5
_output_shapes#
!:                  А2
Conv1D_3_256/mulР
Conv1D_3_256/IdentityIdentityConv1D_3_256/mul:z:0*
T0*5
_output_shapes#
!:                  А2
Conv1D_3_256/IdentityЙ
Conv1D_3_256/IdentityN	IdentityNConv1D_3_256/mul:z:0Conv1D_3_256/BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1294350250*V
_output_shapesD
B:                  А:                  А2
Conv1D_3_256/IdentityN╞
)Batch_Norm_7_256/batchnorm/ReadVariableOpReadVariableOp2batch_norm_7_256_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)Batch_Norm_7_256/batchnorm/ReadVariableOpЙ
 Batch_Norm_7_256/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2"
 Batch_Norm_7_256/batchnorm/add/y═
Batch_Norm_7_256/batchnorm/addAddV21Batch_Norm_7_256/batchnorm/ReadVariableOp:value:0)Batch_Norm_7_256/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2 
Batch_Norm_7_256/batchnorm/addЧ
 Batch_Norm_7_256/batchnorm/RsqrtRsqrt"Batch_Norm_7_256/batchnorm/add:z:0*
T0*
_output_shapes	
:А2"
 Batch_Norm_7_256/batchnorm/Rsqrt╥
-Batch_Norm_7_256/batchnorm/mul/ReadVariableOpReadVariableOp6batch_norm_7_256_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-Batch_Norm_7_256/batchnorm/mul/ReadVariableOp╩
Batch_Norm_7_256/batchnorm/mulMul$Batch_Norm_7_256/batchnorm/Rsqrt:y:05Batch_Norm_7_256/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2 
Batch_Norm_7_256/batchnorm/mul╨
 Batch_Norm_7_256/batchnorm/mul_1MulConv1D_7_256/IdentityN:output:0"Batch_Norm_7_256/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2"
 Batch_Norm_7_256/batchnorm/mul_1╠
+Batch_Norm_7_256/batchnorm/ReadVariableOp_1ReadVariableOp4batch_norm_7_256_batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02-
+Batch_Norm_7_256/batchnorm/ReadVariableOp_1╩
 Batch_Norm_7_256/batchnorm/mul_2Mul3Batch_Norm_7_256/batchnorm/ReadVariableOp_1:value:0"Batch_Norm_7_256/batchnorm/mul:z:0*
T0*
_output_shapes	
:А2"
 Batch_Norm_7_256/batchnorm/mul_2╠
+Batch_Norm_7_256/batchnorm/ReadVariableOp_2ReadVariableOp4batch_norm_7_256_batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02-
+Batch_Norm_7_256/batchnorm/ReadVariableOp_2╚
Batch_Norm_7_256/batchnorm/subSub3Batch_Norm_7_256/batchnorm/ReadVariableOp_2:value:0$Batch_Norm_7_256/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2 
Batch_Norm_7_256/batchnorm/sub╫
 Batch_Norm_7_256/batchnorm/add_1AddV2$Batch_Norm_7_256/batchnorm/mul_1:z:0"Batch_Norm_7_256/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2"
 Batch_Norm_7_256/batchnorm/add_1╞
)Batch_Norm_5_256/batchnorm/ReadVariableOpReadVariableOp2batch_norm_5_256_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)Batch_Norm_5_256/batchnorm/ReadVariableOpЙ
 Batch_Norm_5_256/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2"
 Batch_Norm_5_256/batchnorm/add/y═
Batch_Norm_5_256/batchnorm/addAddV21Batch_Norm_5_256/batchnorm/ReadVariableOp:value:0)Batch_Norm_5_256/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2 
Batch_Norm_5_256/batchnorm/addЧ
 Batch_Norm_5_256/batchnorm/RsqrtRsqrt"Batch_Norm_5_256/batchnorm/add:z:0*
T0*
_output_shapes	
:А2"
 Batch_Norm_5_256/batchnorm/Rsqrt╥
-Batch_Norm_5_256/batchnorm/mul/ReadVariableOpReadVariableOp6batch_norm_5_256_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-Batch_Norm_5_256/batchnorm/mul/ReadVariableOp╩
Batch_Norm_5_256/batchnorm/mulMul$Batch_Norm_5_256/batchnorm/Rsqrt:y:05Batch_Norm_5_256/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2 
Batch_Norm_5_256/batchnorm/mul╨
 Batch_Norm_5_256/batchnorm/mul_1MulConv1D_5_256/IdentityN:output:0"Batch_Norm_5_256/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2"
 Batch_Norm_5_256/batchnorm/mul_1╠
+Batch_Norm_5_256/batchnorm/ReadVariableOp_1ReadVariableOp4batch_norm_5_256_batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02-
+Batch_Norm_5_256/batchnorm/ReadVariableOp_1╩
 Batch_Norm_5_256/batchnorm/mul_2Mul3Batch_Norm_5_256/batchnorm/ReadVariableOp_1:value:0"Batch_Norm_5_256/batchnorm/mul:z:0*
T0*
_output_shapes	
:А2"
 Batch_Norm_5_256/batchnorm/mul_2╠
+Batch_Norm_5_256/batchnorm/ReadVariableOp_2ReadVariableOp4batch_norm_5_256_batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02-
+Batch_Norm_5_256/batchnorm/ReadVariableOp_2╚
Batch_Norm_5_256/batchnorm/subSub3Batch_Norm_5_256/batchnorm/ReadVariableOp_2:value:0$Batch_Norm_5_256/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2 
Batch_Norm_5_256/batchnorm/sub╫
 Batch_Norm_5_256/batchnorm/add_1AddV2$Batch_Norm_5_256/batchnorm/mul_1:z:0"Batch_Norm_5_256/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2"
 Batch_Norm_5_256/batchnorm/add_1╞
)Batch_Norm_3_256/batchnorm/ReadVariableOpReadVariableOp2batch_norm_3_256_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)Batch_Norm_3_256/batchnorm/ReadVariableOpЙ
 Batch_Norm_3_256/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2"
 Batch_Norm_3_256/batchnorm/add/y═
Batch_Norm_3_256/batchnorm/addAddV21Batch_Norm_3_256/batchnorm/ReadVariableOp:value:0)Batch_Norm_3_256/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2 
Batch_Norm_3_256/batchnorm/addЧ
 Batch_Norm_3_256/batchnorm/RsqrtRsqrt"Batch_Norm_3_256/batchnorm/add:z:0*
T0*
_output_shapes	
:А2"
 Batch_Norm_3_256/batchnorm/Rsqrt╥
-Batch_Norm_3_256/batchnorm/mul/ReadVariableOpReadVariableOp6batch_norm_3_256_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-Batch_Norm_3_256/batchnorm/mul/ReadVariableOp╩
Batch_Norm_3_256/batchnorm/mulMul$Batch_Norm_3_256/batchnorm/Rsqrt:y:05Batch_Norm_3_256/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2 
Batch_Norm_3_256/batchnorm/mul╨
 Batch_Norm_3_256/batchnorm/mul_1MulConv1D_3_256/IdentityN:output:0"Batch_Norm_3_256/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2"
 Batch_Norm_3_256/batchnorm/mul_1╠
+Batch_Norm_3_256/batchnorm/ReadVariableOp_1ReadVariableOp4batch_norm_3_256_batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02-
+Batch_Norm_3_256/batchnorm/ReadVariableOp_1╩
 Batch_Norm_3_256/batchnorm/mul_2Mul3Batch_Norm_3_256/batchnorm/ReadVariableOp_1:value:0"Batch_Norm_3_256/batchnorm/mul:z:0*
T0*
_output_shapes	
:А2"
 Batch_Norm_3_256/batchnorm/mul_2╠
+Batch_Norm_3_256/batchnorm/ReadVariableOp_2ReadVariableOp4batch_norm_3_256_batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02-
+Batch_Norm_3_256/batchnorm/ReadVariableOp_2╚
Batch_Norm_3_256/batchnorm/subSub3Batch_Norm_3_256/batchnorm/ReadVariableOp_2:value:0$Batch_Norm_3_256/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2 
Batch_Norm_3_256/batchnorm/sub╫
 Batch_Norm_3_256/batchnorm/add_1AddV2$Batch_Norm_3_256/batchnorm/mul_1:z:0"Batch_Norm_3_256/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2"
 Batch_Norm_3_256/batchnorm/add_1Ж
 Max_Pool_3/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2"
 Max_Pool_3/Max/reduction_indicesл
Max_Pool_3/MaxMax$Batch_Norm_3_256/batchnorm/add_1:z:0)Max_Pool_3/Max/reduction_indices:output:0*
T0*(
_output_shapes
:         А2
Max_Pool_3/MaxЖ
 Max_Pool_5/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2"
 Max_Pool_5/Max/reduction_indicesл
Max_Pool_5/MaxMax$Batch_Norm_5_256/batchnorm/add_1:z:0)Max_Pool_5/Max/reduction_indices:output:0*
T0*(
_output_shapes
:         А2
Max_Pool_5/MaxЖ
 Max_Pool_7/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2"
 Max_Pool_7/Max/reduction_indicesл
Max_Pool_7/MaxMax$Batch_Norm_7_256/batchnorm/add_1:z:0)Max_Pool_7/Max/reduction_indices:output:0*
T0*(
_output_shapes
:         А2
Max_Pool_7/Max~
Pool_Concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
Pool_Concatenate/concat/axisь
Pool_Concatenate/concatConcatV2Max_Pool_3/Max:output:0Max_Pool_5/Max:output:0Max_Pool_7/Max:output:0%Pool_Concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:         А2
Pool_Concatenate/concatЭ
Concatenate_Dropout/IdentityIdentity Pool_Concatenate/concat:output:0*
T0*(
_output_shapes
:         А2
Concatenate_Dropout/Identityп
 Prediction/MatMul/ReadVariableOpReadVariableOp)prediction_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02"
 Prediction/MatMul/ReadVariableOp│
Prediction/MatMulMatMul%Concatenate_Dropout/Identity:output:0(Prediction/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
Prediction/MatMulн
!Prediction/BiasAdd/ReadVariableOpReadVariableOp*prediction_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!Prediction/BiasAdd/ReadVariableOpн
Prediction/BiasAddBiasAddPrediction/MatMul:product:0)Prediction/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
Prediction/BiasAddВ
Prediction/SigmoidSigmoidPrediction/BiasAdd:output:0*
T0*'
_output_shapes
:         2
Prediction/SigmoidС
IdentityIdentityPrediction/Sigmoid:y:0*^Batch_Norm_3_128/batchnorm/ReadVariableOp,^Batch_Norm_3_128/batchnorm/ReadVariableOp_1,^Batch_Norm_3_128/batchnorm/ReadVariableOp_2.^Batch_Norm_3_128/batchnorm/mul/ReadVariableOp*^Batch_Norm_3_256/batchnorm/ReadVariableOp,^Batch_Norm_3_256/batchnorm/ReadVariableOp_1,^Batch_Norm_3_256/batchnorm/ReadVariableOp_2.^Batch_Norm_3_256/batchnorm/mul/ReadVariableOp*^Batch_Norm_5_128/batchnorm/ReadVariableOp,^Batch_Norm_5_128/batchnorm/ReadVariableOp_1,^Batch_Norm_5_128/batchnorm/ReadVariableOp_2.^Batch_Norm_5_128/batchnorm/mul/ReadVariableOp*^Batch_Norm_5_256/batchnorm/ReadVariableOp,^Batch_Norm_5_256/batchnorm/ReadVariableOp_1,^Batch_Norm_5_256/batchnorm/ReadVariableOp_2.^Batch_Norm_5_256/batchnorm/mul/ReadVariableOp*^Batch_Norm_7_128/batchnorm/ReadVariableOp,^Batch_Norm_7_128/batchnorm/ReadVariableOp_1,^Batch_Norm_7_128/batchnorm/ReadVariableOp_2.^Batch_Norm_7_128/batchnorm/mul/ReadVariableOp*^Batch_Norm_7_256/batchnorm/ReadVariableOp,^Batch_Norm_7_256/batchnorm/ReadVariableOp_1,^Batch_Norm_7_256/batchnorm/ReadVariableOp_2.^Batch_Norm_7_256/batchnorm/mul/ReadVariableOp$^Conv1D_3_128/BiasAdd/ReadVariableOp0^Conv1D_3_128/conv1d/ExpandDims_1/ReadVariableOp$^Conv1D_3_256/BiasAdd/ReadVariableOp0^Conv1D_3_256/conv1d/ExpandDims_1/ReadVariableOp$^Conv1D_5_128/BiasAdd/ReadVariableOp0^Conv1D_5_128/conv1d/ExpandDims_1/ReadVariableOp$^Conv1D_5_256/BiasAdd/ReadVariableOp0^Conv1D_5_256/conv1d/ExpandDims_1/ReadVariableOp$^Conv1D_7_128/BiasAdd/ReadVariableOp0^Conv1D_7_128/conv1d/ExpandDims_1/ReadVariableOp$^Conv1D_7_256/BiasAdd/ReadVariableOp0^Conv1D_7_256/conv1d/ExpandDims_1/ReadVariableOp"^Prediction/BiasAdd/ReadVariableOp!^Prediction/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*А
_input_shapeso
m:                  А: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2V
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
!:                  А
 
_user_specified_nameinputs
№*
э
P__inference_Batch_Norm_3_256_layer_call_and_return_conditional_losses_1294348405

inputs6
'assignmovingavg_readvariableop_resource:	А8
)assignmovingavg_1_readvariableop_resource:	А4
%batchnorm_mul_readvariableop_resource:	А0
!batchnorm_readvariableop_resource:	А
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesФ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/meanБ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:А2
moments/StopGradient▓
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:                  А2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╖
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/varianceВ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/SqueezeК
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayе
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOpЩ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:А2
AssignMovingAvg/subР
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:А2
AssignMovingAvg/mul┐
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
╫#<2
AssignMovingAvg_1/decayл
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOpб
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:А2
AssignMovingAvg_1/subШ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:А2
AssignMovingAvg_1/mul╔
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
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/add_1Щ
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  А: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
╢
o
Q__inference_Embedding_Dropout_layer_call_and_return_conditional_losses_1294348858

inputs

identity_1h
IdentityIdentityinputs*
T0*5
_output_shapes#
!:                  А2

Identityw

Identity_1IdentityIdentity:output:0*
T0*5
_output_shapes#
!:                  А2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:                  А:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
№*
э
P__inference_Batch_Norm_5_128_layer_call_and_return_conditional_losses_1294348081

inputs6
'assignmovingavg_readvariableop_resource:	А8
)assignmovingavg_1_readvariableop_resource:	А4
%batchnorm_mul_readvariableop_resource:	А0
!batchnorm_readvariableop_resource:	А
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesФ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/meanБ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:А2
moments/StopGradient▓
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:                  А2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╖
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/varianceВ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/SqueezeК
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayе
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOpЩ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:А2
AssignMovingAvg/subР
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:А2
AssignMovingAvg/mul┐
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
╫#<2
AssignMovingAvg_1/decayл
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOpб
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:А2
AssignMovingAvg_1/subШ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:А2
AssignMovingAvg_1/mul╔
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
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/add_1Щ
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  А: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
╝
╘
5__inference_Batch_Norm_5_128_layer_call_fn_1294350861

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_Batch_Norm_5_128_layer_call_and_return_conditional_losses_12943480812
StatefulPartitionedCallЬ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
Д
q
S__inference_Concatenate_Dropout_layer_call_and_return_conditional_losses_1294351355

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         А2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Ё
Ш	
1__inference_Tree_Encoder_layer_call_fn_1294349659

inputs
unknown:АА
	unknown_0:	А!
	unknown_1:АА
	unknown_2:	А!
	unknown_3:АА
	unknown_4:	А
	unknown_5:	А
	unknown_6:	А
	unknown_7:	А
	unknown_8:	А
	unknown_9:	А

unknown_10:	А

unknown_11:	А

unknown_12:	А

unknown_13:	А

unknown_14:	А

unknown_15:	А

unknown_16:	А"

unknown_17:АА

unknown_18:	А"

unknown_19:АА

unknown_20:	А"

unknown_21:АА

unknown_22:	А

unknown_23:	А

unknown_24:	А

unknown_25:	А

unknown_26:	А

unknown_27:	А

unknown_28:	А

unknown_29:	А

unknown_30:	А

unknown_31:	А

unknown_32:	А

unknown_33:	А

unknown_34:	А

unknown_35:	А

unknown_36:
identityИвStatefulPartitionedCallт
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
:         *<
_read_only_resource_inputs
	
 #$%&*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_Tree_Encoder_layer_call_and_return_conditional_losses_12943494992
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*А
_input_shapeso
m:                  А: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameInputs
№
а
L__inference_Conv1D_5_256_layer_call_and_return_conditional_losses_1294351055

inputsC
+conv1d_expanddims_1_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А

identity_1ИвBiasAdd/ReadVariableOpв"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimа
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#                  А2
conv1d/ExpandDims║
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:АА*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim╣
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:АА2
conv1d/ExpandDims_1└
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#                  А*
paddingSAME*
strides
2
conv1dЬ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*5
_output_shapes#
!:                  А*
squeeze_dims

¤        2
conv1d/SqueezeН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЦ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:                  А2	
BiasAddo
SigmoidSigmoidBiasAdd:output:0*
T0*5
_output_shapes#
!:                  А2	
Sigmoidp
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*5
_output_shapes#
!:                  А2
muli
IdentityIdentitymul:z:0*
T0*5
_output_shapes#
!:                  А2

Identity╒
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1294351048*V
_output_shapesD
B:                  А:                  А2
	IdentityN╢

Identity_1IdentityIdentityN:output:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*5
_output_shapes#
!:                  А2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:                  А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
№
а
L__inference_Conv1D_7_256_layer_call_and_return_conditional_losses_1294351085

inputsC
+conv1d_expanddims_1_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А

identity_1ИвBiasAdd/ReadVariableOpв"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimа
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#                  А2
conv1d/ExpandDims║
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:АА*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim╣
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:АА2
conv1d/ExpandDims_1└
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#                  А*
paddingSAME*
strides
2
conv1dЬ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*5
_output_shapes#
!:                  А*
squeeze_dims

¤        2
conv1d/SqueezeН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЦ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:                  А2	
BiasAddo
SigmoidSigmoidBiasAdd:output:0*
T0*5
_output_shapes#
!:                  А2	
Sigmoidp
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*5
_output_shapes#
!:                  А2
muli
IdentityIdentitymul:z:0*
T0*5
_output_shapes#
!:                  А2

Identity╒
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1294351078*V
_output_shapesD
B:                  А:                  А2
	IdentityN╢

Identity_1IdentityIdentityN:output:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*5
_output_shapes#
!:                  А2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:                  А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
╛
╘
5__inference_Batch_Norm_3_256_layer_call_fn_1294351098

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_Batch_Norm_3_256_layer_call_and_return_conditional_losses_12943483452
StatefulPartitionedCallЬ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
╛
╘
5__inference_Batch_Norm_7_128_layer_call_fn_1294350928

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_Batch_Norm_7_128_layer_call_and_return_conditional_losses_12943481832
StatefulPartitionedCallЬ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
Ё
Ш	
1__inference_Tree_Encoder_layer_call_fn_1294350102

inputs
unknown:АА
	unknown_0:	А!
	unknown_1:АА
	unknown_2:	А!
	unknown_3:АА
	unknown_4:	А
	unknown_5:	А
	unknown_6:	А
	unknown_7:	А
	unknown_8:	А
	unknown_9:	А

unknown_10:	А

unknown_11:	А

unknown_12:	А

unknown_13:	А

unknown_14:	А

unknown_15:	А

unknown_16:	А"

unknown_17:АА

unknown_18:	А"

unknown_19:АА

unknown_20:	А"

unknown_21:АА

unknown_22:	А

unknown_23:	А

unknown_24:	А

unknown_25:	А

unknown_26:	А

unknown_27:	А

unknown_28:	А

unknown_29:	А

unknown_30:	А

unknown_31:	А

unknown_32:	А

unknown_33:	А

unknown_34:	А

unknown_35:	А

unknown_36:
identityИвStatefulPartitionedCallт
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
:         *<
_read_only_resource_inputs
	
 #$%&*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_Tree_Encoder_layer_call_and_return_conditional_losses_12943494992
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*А
_input_shapeso
m:                  А: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
ё
│
P__inference_Batch_Norm_3_256_layer_call_and_return_conditional_losses_1294348345

inputs0
!batchnorm_readvariableop_resource:	А4
%batchnorm_mul_readvariableop_resource:	А2
#batchnorm_readvariableop_1_resource:	А2
#batchnorm_readvariableop_2_resource:	А
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/add_1щ
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  А: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
яn
█
L__inference_Tree_Encoder_layer_call_and_return_conditional_losses_1294349857

inputs/
conv1d_7_128_1294349762:АА&
conv1d_7_128_1294349764:	А/
conv1d_5_128_1294349767:АА&
conv1d_5_128_1294349769:	А/
conv1d_3_128_1294349772:АА&
conv1d_3_128_1294349774:	А*
batch_norm_7_128_1294349777:	А*
batch_norm_7_128_1294349779:	А*
batch_norm_7_128_1294349781:	А*
batch_norm_7_128_1294349783:	А*
batch_norm_5_128_1294349786:	А*
batch_norm_5_128_1294349788:	А*
batch_norm_5_128_1294349790:	А*
batch_norm_5_128_1294349792:	А*
batch_norm_3_128_1294349795:	А*
batch_norm_3_128_1294349797:	А*
batch_norm_3_128_1294349799:	А*
batch_norm_3_128_1294349801:	А/
conv1d_7_256_1294349804:АА&
conv1d_7_256_1294349806:	А/
conv1d_5_256_1294349809:АА&
conv1d_5_256_1294349811:	А/
conv1d_3_256_1294349814:АА&
conv1d_3_256_1294349816:	А*
batch_norm_7_256_1294349819:	А*
batch_norm_7_256_1294349821:	А*
batch_norm_7_256_1294349823:	А*
batch_norm_7_256_1294349825:	А*
batch_norm_5_256_1294349828:	А*
batch_norm_5_256_1294349830:	А*
batch_norm_5_256_1294349832:	А*
batch_norm_5_256_1294349834:	А*
batch_norm_3_256_1294349837:	А*
batch_norm_3_256_1294349839:	А*
batch_norm_3_256_1294349841:	А*
batch_norm_3_256_1294349843:	А(
prediction_1294349851:	А#
prediction_1294349853:
identityИв(Batch_Norm_3_128/StatefulPartitionedCallв(Batch_Norm_3_256/StatefulPartitionedCallв(Batch_Norm_5_128/StatefulPartitionedCallв(Batch_Norm_5_256/StatefulPartitionedCallв(Batch_Norm_7_128/StatefulPartitionedCallв(Batch_Norm_7_256/StatefulPartitionedCallв+Concatenate_Dropout/StatefulPartitionedCallв$Conv1D_3_128/StatefulPartitionedCallв$Conv1D_3_256/StatefulPartitionedCallв$Conv1D_5_128/StatefulPartitionedCallв$Conv1D_5_256/StatefulPartitionedCallв$Conv1D_7_128/StatefulPartitionedCallв$Conv1D_7_256/StatefulPartitionedCallв)Embedding_Dropout/StatefulPartitionedCallв"Prediction/StatefulPartitionedCallЬ
)Embedding_Dropout/StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_Embedding_Dropout_layer_call_and_return_conditional_losses_12943493142+
)Embedding_Dropout/StatefulPartitionedCallё
$Conv1D_7_128/StatefulPartitionedCallStatefulPartitionedCall2Embedding_Dropout/StatefulPartitionedCall:output:0conv1d_7_128_1294349762conv1d_7_128_1294349764*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_Conv1D_7_128_layer_call_and_return_conditional_losses_12943488812&
$Conv1D_7_128/StatefulPartitionedCallё
$Conv1D_5_128/StatefulPartitionedCallStatefulPartitionedCall2Embedding_Dropout/StatefulPartitionedCall:output:0conv1d_5_128_1294349767conv1d_5_128_1294349769*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_Conv1D_5_128_layer_call_and_return_conditional_losses_12943489082&
$Conv1D_5_128/StatefulPartitionedCallё
$Conv1D_3_128/StatefulPartitionedCallStatefulPartitionedCall2Embedding_Dropout/StatefulPartitionedCall:output:0conv1d_3_128_1294349772conv1d_3_128_1294349774*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_Conv1D_3_128_layer_call_and_return_conditional_losses_12943489352&
$Conv1D_3_128/StatefulPartitionedCall╝
(Batch_Norm_7_128/StatefulPartitionedCallStatefulPartitionedCall-Conv1D_7_128/StatefulPartitionedCall:output:0batch_norm_7_128_1294349777batch_norm_7_128_1294349779batch_norm_7_128_1294349781batch_norm_7_128_1294349783*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_Batch_Norm_7_128_layer_call_and_return_conditional_losses_12943482432*
(Batch_Norm_7_128/StatefulPartitionedCall╝
(Batch_Norm_5_128/StatefulPartitionedCallStatefulPartitionedCall-Conv1D_5_128/StatefulPartitionedCall:output:0batch_norm_5_128_1294349786batch_norm_5_128_1294349788batch_norm_5_128_1294349790batch_norm_5_128_1294349792*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_Batch_Norm_5_128_layer_call_and_return_conditional_losses_12943480812*
(Batch_Norm_5_128/StatefulPartitionedCall╝
(Batch_Norm_3_128/StatefulPartitionedCallStatefulPartitionedCall-Conv1D_3_128/StatefulPartitionedCall:output:0batch_norm_3_128_1294349795batch_norm_3_128_1294349797batch_norm_3_128_1294349799batch_norm_3_128_1294349801*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_Batch_Norm_3_128_layer_call_and_return_conditional_losses_12943479192*
(Batch_Norm_3_128/StatefulPartitionedCallЁ
$Conv1D_7_256/StatefulPartitionedCallStatefulPartitionedCall1Batch_Norm_7_128/StatefulPartitionedCall:output:0conv1d_7_256_1294349804conv1d_7_256_1294349806*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_Conv1D_7_256_layer_call_and_return_conditional_losses_12943489892&
$Conv1D_7_256/StatefulPartitionedCallЁ
$Conv1D_5_256/StatefulPartitionedCallStatefulPartitionedCall1Batch_Norm_5_128/StatefulPartitionedCall:output:0conv1d_5_256_1294349809conv1d_5_256_1294349811*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_Conv1D_5_256_layer_call_and_return_conditional_losses_12943490162&
$Conv1D_5_256/StatefulPartitionedCallЁ
$Conv1D_3_256/StatefulPartitionedCallStatefulPartitionedCall1Batch_Norm_3_128/StatefulPartitionedCall:output:0conv1d_3_256_1294349814conv1d_3_256_1294349816*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_Conv1D_3_256_layer_call_and_return_conditional_losses_12943490432&
$Conv1D_3_256/StatefulPartitionedCall╝
(Batch_Norm_7_256/StatefulPartitionedCallStatefulPartitionedCall-Conv1D_7_256/StatefulPartitionedCall:output:0batch_norm_7_256_1294349819batch_norm_7_256_1294349821batch_norm_7_256_1294349823batch_norm_7_256_1294349825*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_Batch_Norm_7_256_layer_call_and_return_conditional_losses_12943487292*
(Batch_Norm_7_256/StatefulPartitionedCall╝
(Batch_Norm_5_256/StatefulPartitionedCallStatefulPartitionedCall-Conv1D_5_256/StatefulPartitionedCall:output:0batch_norm_5_256_1294349828batch_norm_5_256_1294349830batch_norm_5_256_1294349832batch_norm_5_256_1294349834*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_Batch_Norm_5_256_layer_call_and_return_conditional_losses_12943485672*
(Batch_Norm_5_256/StatefulPartitionedCall╝
(Batch_Norm_3_256/StatefulPartitionedCallStatefulPartitionedCall-Conv1D_3_256/StatefulPartitionedCall:output:0batch_norm_3_256_1294349837batch_norm_3_256_1294349839batch_norm_3_256_1294349841batch_norm_3_256_1294349843*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_Batch_Norm_3_256_layer_call_and_return_conditional_losses_12943484052*
(Batch_Norm_3_256/StatefulPartitionedCallН
Max_Pool_3/PartitionedCallPartitionedCall1Batch_Norm_3_256/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_Max_Pool_3_layer_call_and_return_conditional_losses_12943488142
Max_Pool_3/PartitionedCallН
Max_Pool_5/PartitionedCallPartitionedCall1Batch_Norm_5_256/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_Max_Pool_5_layer_call_and_return_conditional_losses_12943488272
Max_Pool_5/PartitionedCallН
Max_Pool_7/PartitionedCallPartitionedCall1Batch_Norm_7_256/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_Max_Pool_7_layer_call_and_return_conditional_losses_12943488402
Max_Pool_7/PartitionedCall▌
 Pool_Concatenate/PartitionedCallPartitionedCall#Max_Pool_3/PartitionedCall:output:0#Max_Pool_5/PartitionedCall:output:0#Max_Pool_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_Pool_Concatenate_layer_call_and_return_conditional_losses_12943490872"
 Pool_Concatenate/PartitionedCallф
+Concatenate_Dropout/StatefulPartitionedCallStatefulPartitionedCall)Pool_Concatenate/PartitionedCall:output:0*^Embedding_Dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_Concatenate_Dropout_layer_call_and_return_conditional_losses_12943492232-
+Concatenate_Dropout/StatefulPartitionedCall█
"Prediction/StatefulPartitionedCallStatefulPartitionedCall4Concatenate_Dropout/StatefulPartitionedCall:output:0prediction_1294349851prediction_1294349853*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_Prediction_layer_call_and_return_conditional_losses_12943491072$
"Prediction/StatefulPartitionedCallъ
IdentityIdentity+Prediction/StatefulPartitionedCall:output:0)^Batch_Norm_3_128/StatefulPartitionedCall)^Batch_Norm_3_256/StatefulPartitionedCall)^Batch_Norm_5_128/StatefulPartitionedCall)^Batch_Norm_5_256/StatefulPartitionedCall)^Batch_Norm_7_128/StatefulPartitionedCall)^Batch_Norm_7_256/StatefulPartitionedCall,^Concatenate_Dropout/StatefulPartitionedCall%^Conv1D_3_128/StatefulPartitionedCall%^Conv1D_3_256/StatefulPartitionedCall%^Conv1D_5_128/StatefulPartitionedCall%^Conv1D_5_256/StatefulPartitionedCall%^Conv1D_7_128/StatefulPartitionedCall%^Conv1D_7_256/StatefulPartitionedCall*^Embedding_Dropout/StatefulPartitionedCall#^Prediction/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*А
_input_shapeso
m:                  А: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2T
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
!:                  А
 
_user_specified_nameInputs
э
е
1__inference_Conv1D_3_256_layer_call_fn_1294351004

inputs
unknown:АА
	unknown_0:	А
identityИвStatefulPartitionedCallН
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_Conv1D_3_256_layer_call_and_return_conditional_losses_12943490432
StatefulPartitionedCallЬ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:                  А: : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
╝
╘
5__inference_Batch_Norm_7_256_layer_call_fn_1294351271

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_Batch_Norm_7_256_layer_call_and_return_conditional_losses_12943487292
StatefulPartitionedCallЬ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
╠
П	
(__inference_signature_wrapper_1294349940

inputs
unknown:АА
	unknown_0:	А!
	unknown_1:АА
	unknown_2:	А!
	unknown_3:АА
	unknown_4:	А
	unknown_5:	А
	unknown_6:	А
	unknown_7:	А
	unknown_8:	А
	unknown_9:	А

unknown_10:	А

unknown_11:	А

unknown_12:	А

unknown_13:	А

unknown_14:	А

unknown_15:	А

unknown_16:	А"

unknown_17:АА

unknown_18:	А"

unknown_19:АА

unknown_20:	А"

unknown_21:АА

unknown_22:	А

unknown_23:	А

unknown_24:	А

unknown_25:	А

unknown_26:	А

unknown_27:	А

unknown_28:	А

unknown_29:	А

unknown_30:	А

unknown_31:	А

unknown_32:	А

unknown_33:	А

unknown_34:	А

unknown_35:	А

unknown_36:
identityИвStatefulPartitionedCall╟
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
:         *H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8В *.
f)R'
%__inference__wrapped_model_12943478352
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*А
_input_shapeso
m:                  А: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameInputs
ё
│
P__inference_Batch_Norm_7_256_layer_call_and_return_conditional_losses_1294348669

inputs0
!batchnorm_readvariableop_resource:	А4
%batchnorm_mul_readvariableop_resource:	А2
#batchnorm_readvariableop_1_resource:	А2
#batchnorm_readvariableop_2_resource:	А
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/add_1щ
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  А: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
╛
╘
5__inference_Batch_Norm_3_128_layer_call_fn_1294350768

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_Batch_Norm_3_128_layer_call_and_return_conditional_losses_12943478592
StatefulPartitionedCallЬ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
╕

№
J__inference_Prediction_layer_call_and_return_conditional_losses_1294351387

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         2	
SigmoidР
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
ё
│
P__inference_Batch_Norm_7_256_layer_call_and_return_conditional_losses_1294351291

inputs0
!batchnorm_readvariableop_resource:	А4
%batchnorm_mul_readvariableop_resource:	А2
#batchnorm_readvariableop_1_resource:	А2
#batchnorm_readvariableop_2_resource:	А
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/add_1щ
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  А: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
Чk
Б
L__inference_Tree_Encoder_layer_call_and_return_conditional_losses_1294349114

inputs/
conv1d_7_128_1294348882:АА&
conv1d_7_128_1294348884:	А/
conv1d_5_128_1294348909:АА&
conv1d_5_128_1294348911:	А/
conv1d_3_128_1294348936:АА&
conv1d_3_128_1294348938:	А*
batch_norm_7_128_1294348941:	А*
batch_norm_7_128_1294348943:	А*
batch_norm_7_128_1294348945:	А*
batch_norm_7_128_1294348947:	А*
batch_norm_5_128_1294348950:	А*
batch_norm_5_128_1294348952:	А*
batch_norm_5_128_1294348954:	А*
batch_norm_5_128_1294348956:	А*
batch_norm_3_128_1294348959:	А*
batch_norm_3_128_1294348961:	А*
batch_norm_3_128_1294348963:	А*
batch_norm_3_128_1294348965:	А/
conv1d_7_256_1294348990:АА&
conv1d_7_256_1294348992:	А/
conv1d_5_256_1294349017:АА&
conv1d_5_256_1294349019:	А/
conv1d_3_256_1294349044:АА&
conv1d_3_256_1294349046:	А*
batch_norm_7_256_1294349049:	А*
batch_norm_7_256_1294349051:	А*
batch_norm_7_256_1294349053:	А*
batch_norm_7_256_1294349055:	А*
batch_norm_5_256_1294349058:	А*
batch_norm_5_256_1294349060:	А*
batch_norm_5_256_1294349062:	А*
batch_norm_5_256_1294349064:	А*
batch_norm_3_256_1294349067:	А*
batch_norm_3_256_1294349069:	А*
batch_norm_3_256_1294349071:	А*
batch_norm_3_256_1294349073:	А(
prediction_1294349108:	А#
prediction_1294349110:
identityИв(Batch_Norm_3_128/StatefulPartitionedCallв(Batch_Norm_3_256/StatefulPartitionedCallв(Batch_Norm_5_128/StatefulPartitionedCallв(Batch_Norm_5_256/StatefulPartitionedCallв(Batch_Norm_7_128/StatefulPartitionedCallв(Batch_Norm_7_256/StatefulPartitionedCallв$Conv1D_3_128/StatefulPartitionedCallв$Conv1D_3_256/StatefulPartitionedCallв$Conv1D_5_128/StatefulPartitionedCallв$Conv1D_5_256/StatefulPartitionedCallв$Conv1D_7_128/StatefulPartitionedCallв$Conv1D_7_256/StatefulPartitionedCallв"Prediction/StatefulPartitionedCallД
!Embedding_Dropout/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_Embedding_Dropout_layer_call_and_return_conditional_losses_12943488582#
!Embedding_Dropout/PartitionedCallщ
$Conv1D_7_128/StatefulPartitionedCallStatefulPartitionedCall*Embedding_Dropout/PartitionedCall:output:0conv1d_7_128_1294348882conv1d_7_128_1294348884*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_Conv1D_7_128_layer_call_and_return_conditional_losses_12943488812&
$Conv1D_7_128/StatefulPartitionedCallщ
$Conv1D_5_128/StatefulPartitionedCallStatefulPartitionedCall*Embedding_Dropout/PartitionedCall:output:0conv1d_5_128_1294348909conv1d_5_128_1294348911*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_Conv1D_5_128_layer_call_and_return_conditional_losses_12943489082&
$Conv1D_5_128/StatefulPartitionedCallщ
$Conv1D_3_128/StatefulPartitionedCallStatefulPartitionedCall*Embedding_Dropout/PartitionedCall:output:0conv1d_3_128_1294348936conv1d_3_128_1294348938*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_Conv1D_3_128_layer_call_and_return_conditional_losses_12943489352&
$Conv1D_3_128/StatefulPartitionedCall╛
(Batch_Norm_7_128/StatefulPartitionedCallStatefulPartitionedCall-Conv1D_7_128/StatefulPartitionedCall:output:0batch_norm_7_128_1294348941batch_norm_7_128_1294348943batch_norm_7_128_1294348945batch_norm_7_128_1294348947*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_Batch_Norm_7_128_layer_call_and_return_conditional_losses_12943481832*
(Batch_Norm_7_128/StatefulPartitionedCall╛
(Batch_Norm_5_128/StatefulPartitionedCallStatefulPartitionedCall-Conv1D_5_128/StatefulPartitionedCall:output:0batch_norm_5_128_1294348950batch_norm_5_128_1294348952batch_norm_5_128_1294348954batch_norm_5_128_1294348956*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_Batch_Norm_5_128_layer_call_and_return_conditional_losses_12943480212*
(Batch_Norm_5_128/StatefulPartitionedCall╛
(Batch_Norm_3_128/StatefulPartitionedCallStatefulPartitionedCall-Conv1D_3_128/StatefulPartitionedCall:output:0batch_norm_3_128_1294348959batch_norm_3_128_1294348961batch_norm_3_128_1294348963batch_norm_3_128_1294348965*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_Batch_Norm_3_128_layer_call_and_return_conditional_losses_12943478592*
(Batch_Norm_3_128/StatefulPartitionedCallЁ
$Conv1D_7_256/StatefulPartitionedCallStatefulPartitionedCall1Batch_Norm_7_128/StatefulPartitionedCall:output:0conv1d_7_256_1294348990conv1d_7_256_1294348992*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_Conv1D_7_256_layer_call_and_return_conditional_losses_12943489892&
$Conv1D_7_256/StatefulPartitionedCallЁ
$Conv1D_5_256/StatefulPartitionedCallStatefulPartitionedCall1Batch_Norm_5_128/StatefulPartitionedCall:output:0conv1d_5_256_1294349017conv1d_5_256_1294349019*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_Conv1D_5_256_layer_call_and_return_conditional_losses_12943490162&
$Conv1D_5_256/StatefulPartitionedCallЁ
$Conv1D_3_256/StatefulPartitionedCallStatefulPartitionedCall1Batch_Norm_3_128/StatefulPartitionedCall:output:0conv1d_3_256_1294349044conv1d_3_256_1294349046*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_Conv1D_3_256_layer_call_and_return_conditional_losses_12943490432&
$Conv1D_3_256/StatefulPartitionedCall╛
(Batch_Norm_7_256/StatefulPartitionedCallStatefulPartitionedCall-Conv1D_7_256/StatefulPartitionedCall:output:0batch_norm_7_256_1294349049batch_norm_7_256_1294349051batch_norm_7_256_1294349053batch_norm_7_256_1294349055*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_Batch_Norm_7_256_layer_call_and_return_conditional_losses_12943486692*
(Batch_Norm_7_256/StatefulPartitionedCall╛
(Batch_Norm_5_256/StatefulPartitionedCallStatefulPartitionedCall-Conv1D_5_256/StatefulPartitionedCall:output:0batch_norm_5_256_1294349058batch_norm_5_256_1294349060batch_norm_5_256_1294349062batch_norm_5_256_1294349064*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_Batch_Norm_5_256_layer_call_and_return_conditional_losses_12943485072*
(Batch_Norm_5_256/StatefulPartitionedCall╛
(Batch_Norm_3_256/StatefulPartitionedCallStatefulPartitionedCall-Conv1D_3_256/StatefulPartitionedCall:output:0batch_norm_3_256_1294349067batch_norm_3_256_1294349069batch_norm_3_256_1294349071batch_norm_3_256_1294349073*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_Batch_Norm_3_256_layer_call_and_return_conditional_losses_12943483452*
(Batch_Norm_3_256/StatefulPartitionedCallН
Max_Pool_3/PartitionedCallPartitionedCall1Batch_Norm_3_256/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_Max_Pool_3_layer_call_and_return_conditional_losses_12943488142
Max_Pool_3/PartitionedCallН
Max_Pool_5/PartitionedCallPartitionedCall1Batch_Norm_5_256/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_Max_Pool_5_layer_call_and_return_conditional_losses_12943488272
Max_Pool_5/PartitionedCallН
Max_Pool_7/PartitionedCallPartitionedCall1Batch_Norm_7_256/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_Max_Pool_7_layer_call_and_return_conditional_losses_12943488402
Max_Pool_7/PartitionedCall▌
 Pool_Concatenate/PartitionedCallPartitionedCall#Max_Pool_3/PartitionedCall:output:0#Max_Pool_5/PartitionedCall:output:0#Max_Pool_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_Pool_Concatenate_layer_call_and_return_conditional_losses_12943490872"
 Pool_Concatenate/PartitionedCallа
#Concatenate_Dropout/PartitionedCallPartitionedCall)Pool_Concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_Concatenate_Dropout_layer_call_and_return_conditional_losses_12943490942%
#Concatenate_Dropout/PartitionedCall╙
"Prediction/StatefulPartitionedCallStatefulPartitionedCall,Concatenate_Dropout/PartitionedCall:output:0prediction_1294349108prediction_1294349110*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_Prediction_layer_call_and_return_conditional_losses_12943491072$
"Prediction/StatefulPartitionedCallР
IdentityIdentity+Prediction/StatefulPartitionedCall:output:0)^Batch_Norm_3_128/StatefulPartitionedCall)^Batch_Norm_3_256/StatefulPartitionedCall)^Batch_Norm_5_128/StatefulPartitionedCall)^Batch_Norm_5_256/StatefulPartitionedCall)^Batch_Norm_7_128/StatefulPartitionedCall)^Batch_Norm_7_256/StatefulPartitionedCall%^Conv1D_3_128/StatefulPartitionedCall%^Conv1D_3_256/StatefulPartitionedCall%^Conv1D_5_128/StatefulPartitionedCall%^Conv1D_5_256/StatefulPartitionedCall%^Conv1D_7_128/StatefulPartitionedCall%^Conv1D_7_256/StatefulPartitionedCall#^Prediction/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*А
_input_shapeso
m:                  А: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2T
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
!:                  А
 
_user_specified_nameinputs
№*
э
P__inference_Batch_Norm_7_256_layer_call_and_return_conditional_losses_1294351325

inputs6
'assignmovingavg_readvariableop_resource:	А8
)assignmovingavg_1_readvariableop_resource:	А4
%batchnorm_mul_readvariableop_resource:	А0
!batchnorm_readvariableop_resource:	А
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesФ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/meanБ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:А2
moments/StopGradient▓
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:                  А2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╖
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/varianceВ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/SqueezeК
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayе
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOpЩ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:А2
AssignMovingAvg/subР
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:А2
AssignMovingAvg/mul┐
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
╫#<2
AssignMovingAvg_1/decayл
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOpб
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:А2
AssignMovingAvg_1/subШ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:А2
AssignMovingAvg_1/mul╔
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
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/add_1Щ
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  А: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
к
p
Q__inference_Embedding_Dropout_layer_call_and_return_conditional_losses_1294350665

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?2
dropout/ConstБ
dropout/MulMulinputsdropout/Const:output:0*
T0*5
_output_shapes#
!:                  А2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┬
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*5
_output_shapes#
!:                  А*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
dropout/GreaterEqual/y╠
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*5
_output_shapes#
!:                  А2
dropout/GreaterEqualН
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*5
_output_shapes#
!:                  А2
dropout/CastИ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*5
_output_shapes#
!:                  А2
dropout/Mul_1s
IdentityIdentitydropout/Mul_1:z:0*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:                  А:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
э
е
1__inference_Conv1D_3_128_layer_call_fn_1294350674

inputs
unknown:АА
	unknown_0:	А
identityИвStatefulPartitionedCallН
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_Conv1D_3_128_layer_call_and_return_conditional_losses_12943489352
StatefulPartitionedCallЬ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:                  А: : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
э
q
8__inference_Concatenate_Dropout_layer_call_fn_1294351350

inputs
identityИвStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_Concatenate_Dropout_layer_call_and_return_conditional_losses_12943492232
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
┬
r
S__inference_Concatenate_Dropout_layer_call_and_return_conditional_losses_1294351367

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         А2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╡
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
№
а
L__inference_Conv1D_3_128_layer_call_and_return_conditional_losses_1294350695

inputsC
+conv1d_expanddims_1_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А

identity_1ИвBiasAdd/ReadVariableOpв"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimа
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#                  А2
conv1d/ExpandDims║
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:АА*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim╣
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:АА2
conv1d/ExpandDims_1└
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#                  А*
paddingSAME*
strides
2
conv1dЬ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*5
_output_shapes#
!:                  А*
squeeze_dims

¤        2
conv1d/SqueezeН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЦ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:                  А2	
BiasAddo
SigmoidSigmoidBiasAdd:output:0*
T0*5
_output_shapes#
!:                  А2	
Sigmoidp
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*5
_output_shapes#
!:                  А2
muli
IdentityIdentitymul:z:0*
T0*5
_output_shapes#
!:                  А2

Identity╒
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1294350688*V
_output_shapesD
B:                  А:                  А2
	IdentityN╢

Identity_1IdentityIdentityN:output:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*5
_output_shapes#
!:                  А2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:                  А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
╝
╘
5__inference_Batch_Norm_7_128_layer_call_fn_1294350941

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_Batch_Norm_7_128_layer_call_and_return_conditional_losses_12943482432
StatefulPartitionedCallЬ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
№*
э
P__inference_Batch_Norm_5_256_layer_call_and_return_conditional_losses_1294351245

inputs6
'assignmovingavg_readvariableop_resource:	А8
)assignmovingavg_1_readvariableop_resource:	А4
%batchnorm_mul_readvariableop_resource:	А0
!batchnorm_readvariableop_resource:	А
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesФ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/meanБ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:А2
moments/StopGradient▓
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:                  А2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╖
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/varianceВ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/SqueezeК
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayе
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOpЩ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:А2
AssignMovingAvg/subР
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:А2
AssignMovingAvg/mul┐
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
╫#<2
AssignMovingAvg_1/decayл
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOpб
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:А2
AssignMovingAvg_1/subШ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:А2
AssignMovingAvg_1/mul╔
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
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/add_1Щ
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  А: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
ь
И
P__inference_Pool_Concatenate_layer_call_and_return_conditional_losses_1294349087

inputs
inputs_1
inputs_2
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisК
concatConcatV2inputsinputs_1inputs_2concat/axis:output:0*
N*
T0*(
_output_shapes
:         А2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:         А:         А:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs:PL
(
_output_shapes
:         А
 
_user_specified_nameinputs:PL
(
_output_shapes
:         А
 
_user_specified_nameinputs
╝
╘
5__inference_Batch_Norm_5_256_layer_call_fn_1294351191

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_Batch_Norm_5_256_layer_call_and_return_conditional_losses_12943485672
StatefulPartitionedCallЬ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
№*
э
P__inference_Batch_Norm_7_128_layer_call_and_return_conditional_losses_1294348243

inputs6
'assignmovingavg_readvariableop_resource:	А8
)assignmovingavg_1_readvariableop_resource:	А4
%batchnorm_mul_readvariableop_resource:	А0
!batchnorm_readvariableop_resource:	А
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesФ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/meanБ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:А2
moments/StopGradient▓
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:                  А2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╖
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/varianceВ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/SqueezeК
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayе
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOpЩ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:А2
AssignMovingAvg/subР
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:А2
AssignMovingAvg/mul┐
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
╫#<2
AssignMovingAvg_1/decayл
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOpб
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:А2
AssignMovingAvg_1/subШ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:А2
AssignMovingAvg_1/mul╔
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
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/add_1Щ
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  А: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
Й
K
/__inference_Max_Pool_7_layer_call_fn_1294348846

inputs
identity╘
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:                  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_Max_Pool_7_layer_call_and_return_conditional_losses_12943488402
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:                  2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
№
а
L__inference_Conv1D_7_256_layer_call_and_return_conditional_losses_1294348989

inputsC
+conv1d_expanddims_1_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А

identity_1ИвBiasAdd/ReadVariableOpв"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimа
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#                  А2
conv1d/ExpandDims║
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:АА*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim╣
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:АА2
conv1d/ExpandDims_1└
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#                  А*
paddingSAME*
strides
2
conv1dЬ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*5
_output_shapes#
!:                  А*
squeeze_dims

¤        2
conv1d/SqueezeН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЦ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:                  А2	
BiasAddo
SigmoidSigmoidBiasAdd:output:0*
T0*5
_output_shapes#
!:                  А2	
Sigmoidp
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*5
_output_shapes#
!:                  А2
muli
IdentityIdentitymul:z:0*
T0*5
_output_shapes#
!:                  А2

Identity╒
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1294348982*V
_output_shapesD
B:                  А:                  А2
	IdentityN╢

Identity_1IdentityIdentityN:output:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*5
_output_shapes#
!:                  А2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:                  А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
№
а
L__inference_Conv1D_3_256_layer_call_and_return_conditional_losses_1294351025

inputsC
+conv1d_expanddims_1_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А

identity_1ИвBiasAdd/ReadVariableOpв"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimа
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#                  А2
conv1d/ExpandDims║
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:АА*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim╣
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:АА2
conv1d/ExpandDims_1└
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#                  А*
paddingSAME*
strides
2
conv1dЬ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*5
_output_shapes#
!:                  А*
squeeze_dims

¤        2
conv1d/SqueezeН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЦ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:                  А2	
BiasAddo
SigmoidSigmoidBiasAdd:output:0*
T0*5
_output_shapes#
!:                  А2	
Sigmoidp
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*5
_output_shapes#
!:                  А2
muli
IdentityIdentitymul:z:0*
T0*5
_output_shapes#
!:                  А2

Identity╒
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1294351018*V
_output_shapesD
B:                  А:                  А2
	IdentityN╢

Identity_1IdentityIdentityN:output:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*5
_output_shapes#
!:                  А2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:                  А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
№*
э
P__inference_Batch_Norm_3_256_layer_call_and_return_conditional_losses_1294351165

inputs6
'assignmovingavg_readvariableop_resource:	А8
)assignmovingavg_1_readvariableop_resource:	А4
%batchnorm_mul_readvariableop_resource:	А0
!batchnorm_readvariableop_resource:	А
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesФ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/meanБ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:А2
moments/StopGradient▓
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:                  А2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╖
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/varianceВ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/SqueezeК
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayе
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOpЩ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:А2
AssignMovingAvg/subР
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:А2
AssignMovingAvg/mul┐
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
╫#<2
AssignMovingAvg_1/decayл
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOpб
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:А2
AssignMovingAvg_1/subШ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:А2
AssignMovingAvg_1/mul╔
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
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/add_1Щ
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  А: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
ё
│
P__inference_Batch_Norm_3_128_layer_call_and_return_conditional_losses_1294347859

inputs0
!batchnorm_readvariableop_resource:	А4
%batchnorm_mul_readvariableop_resource:	А2
#batchnorm_readvariableop_1_resource:	А2
#batchnorm_readvariableop_2_resource:	А
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/add_1щ
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  А: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
э
е
1__inference_Conv1D_5_256_layer_call_fn_1294351034

inputs
unknown:АА
	unknown_0:	А
identityИвStatefulPartitionedCallН
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_Conv1D_5_256_layer_call_and_return_conditional_losses_12943490162
StatefulPartitionedCallЬ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:                  А: : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
с
T
8__inference_Concatenate_Dropout_layer_call_fn_1294351345

inputs
identity╒
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_Concatenate_Dropout_layer_call_and_return_conditional_losses_12943490942
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
№*
э
P__inference_Batch_Norm_7_128_layer_call_and_return_conditional_losses_1294350995

inputs6
'assignmovingavg_readvariableop_resource:	А8
)assignmovingavg_1_readvariableop_resource:	А4
%batchnorm_mul_readvariableop_resource:	А0
!batchnorm_readvariableop_resource:	А
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesФ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/meanБ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:А2
moments/StopGradient▓
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:                  А2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╖
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/varianceВ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/SqueezeК
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayе
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOpЩ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:А2
AssignMovingAvg/subР
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:А2
AssignMovingAvg/mul┐
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
╫#<2
AssignMovingAvg_1/decayл
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOpб
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:А2
AssignMovingAvg_1/subШ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:А2
AssignMovingAvg_1/mul╔
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
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/add_1Щ
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  А: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
Чk
Б
L__inference_Tree_Encoder_layer_call_and_return_conditional_losses_1294349758

inputs/
conv1d_7_128_1294349663:АА&
conv1d_7_128_1294349665:	А/
conv1d_5_128_1294349668:АА&
conv1d_5_128_1294349670:	А/
conv1d_3_128_1294349673:АА&
conv1d_3_128_1294349675:	А*
batch_norm_7_128_1294349678:	А*
batch_norm_7_128_1294349680:	А*
batch_norm_7_128_1294349682:	А*
batch_norm_7_128_1294349684:	А*
batch_norm_5_128_1294349687:	А*
batch_norm_5_128_1294349689:	А*
batch_norm_5_128_1294349691:	А*
batch_norm_5_128_1294349693:	А*
batch_norm_3_128_1294349696:	А*
batch_norm_3_128_1294349698:	А*
batch_norm_3_128_1294349700:	А*
batch_norm_3_128_1294349702:	А/
conv1d_7_256_1294349705:АА&
conv1d_7_256_1294349707:	А/
conv1d_5_256_1294349710:АА&
conv1d_5_256_1294349712:	А/
conv1d_3_256_1294349715:АА&
conv1d_3_256_1294349717:	А*
batch_norm_7_256_1294349720:	А*
batch_norm_7_256_1294349722:	А*
batch_norm_7_256_1294349724:	А*
batch_norm_7_256_1294349726:	А*
batch_norm_5_256_1294349729:	А*
batch_norm_5_256_1294349731:	А*
batch_norm_5_256_1294349733:	А*
batch_norm_5_256_1294349735:	А*
batch_norm_3_256_1294349738:	А*
batch_norm_3_256_1294349740:	А*
batch_norm_3_256_1294349742:	А*
batch_norm_3_256_1294349744:	А(
prediction_1294349752:	А#
prediction_1294349754:
identityИв(Batch_Norm_3_128/StatefulPartitionedCallв(Batch_Norm_3_256/StatefulPartitionedCallв(Batch_Norm_5_128/StatefulPartitionedCallв(Batch_Norm_5_256/StatefulPartitionedCallв(Batch_Norm_7_128/StatefulPartitionedCallв(Batch_Norm_7_256/StatefulPartitionedCallв$Conv1D_3_128/StatefulPartitionedCallв$Conv1D_3_256/StatefulPartitionedCallв$Conv1D_5_128/StatefulPartitionedCallв$Conv1D_5_256/StatefulPartitionedCallв$Conv1D_7_128/StatefulPartitionedCallв$Conv1D_7_256/StatefulPartitionedCallв"Prediction/StatefulPartitionedCallД
!Embedding_Dropout/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_Embedding_Dropout_layer_call_and_return_conditional_losses_12943488582#
!Embedding_Dropout/PartitionedCallщ
$Conv1D_7_128/StatefulPartitionedCallStatefulPartitionedCall*Embedding_Dropout/PartitionedCall:output:0conv1d_7_128_1294349663conv1d_7_128_1294349665*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_Conv1D_7_128_layer_call_and_return_conditional_losses_12943488812&
$Conv1D_7_128/StatefulPartitionedCallщ
$Conv1D_5_128/StatefulPartitionedCallStatefulPartitionedCall*Embedding_Dropout/PartitionedCall:output:0conv1d_5_128_1294349668conv1d_5_128_1294349670*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_Conv1D_5_128_layer_call_and_return_conditional_losses_12943489082&
$Conv1D_5_128/StatefulPartitionedCallщ
$Conv1D_3_128/StatefulPartitionedCallStatefulPartitionedCall*Embedding_Dropout/PartitionedCall:output:0conv1d_3_128_1294349673conv1d_3_128_1294349675*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_Conv1D_3_128_layer_call_and_return_conditional_losses_12943489352&
$Conv1D_3_128/StatefulPartitionedCall╛
(Batch_Norm_7_128/StatefulPartitionedCallStatefulPartitionedCall-Conv1D_7_128/StatefulPartitionedCall:output:0batch_norm_7_128_1294349678batch_norm_7_128_1294349680batch_norm_7_128_1294349682batch_norm_7_128_1294349684*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_Batch_Norm_7_128_layer_call_and_return_conditional_losses_12943481832*
(Batch_Norm_7_128/StatefulPartitionedCall╛
(Batch_Norm_5_128/StatefulPartitionedCallStatefulPartitionedCall-Conv1D_5_128/StatefulPartitionedCall:output:0batch_norm_5_128_1294349687batch_norm_5_128_1294349689batch_norm_5_128_1294349691batch_norm_5_128_1294349693*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_Batch_Norm_5_128_layer_call_and_return_conditional_losses_12943480212*
(Batch_Norm_5_128/StatefulPartitionedCall╛
(Batch_Norm_3_128/StatefulPartitionedCallStatefulPartitionedCall-Conv1D_3_128/StatefulPartitionedCall:output:0batch_norm_3_128_1294349696batch_norm_3_128_1294349698batch_norm_3_128_1294349700batch_norm_3_128_1294349702*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_Batch_Norm_3_128_layer_call_and_return_conditional_losses_12943478592*
(Batch_Norm_3_128/StatefulPartitionedCallЁ
$Conv1D_7_256/StatefulPartitionedCallStatefulPartitionedCall1Batch_Norm_7_128/StatefulPartitionedCall:output:0conv1d_7_256_1294349705conv1d_7_256_1294349707*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_Conv1D_7_256_layer_call_and_return_conditional_losses_12943489892&
$Conv1D_7_256/StatefulPartitionedCallЁ
$Conv1D_5_256/StatefulPartitionedCallStatefulPartitionedCall1Batch_Norm_5_128/StatefulPartitionedCall:output:0conv1d_5_256_1294349710conv1d_5_256_1294349712*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_Conv1D_5_256_layer_call_and_return_conditional_losses_12943490162&
$Conv1D_5_256/StatefulPartitionedCallЁ
$Conv1D_3_256/StatefulPartitionedCallStatefulPartitionedCall1Batch_Norm_3_128/StatefulPartitionedCall:output:0conv1d_3_256_1294349715conv1d_3_256_1294349717*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_Conv1D_3_256_layer_call_and_return_conditional_losses_12943490432&
$Conv1D_3_256/StatefulPartitionedCall╛
(Batch_Norm_7_256/StatefulPartitionedCallStatefulPartitionedCall-Conv1D_7_256/StatefulPartitionedCall:output:0batch_norm_7_256_1294349720batch_norm_7_256_1294349722batch_norm_7_256_1294349724batch_norm_7_256_1294349726*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_Batch_Norm_7_256_layer_call_and_return_conditional_losses_12943486692*
(Batch_Norm_7_256/StatefulPartitionedCall╛
(Batch_Norm_5_256/StatefulPartitionedCallStatefulPartitionedCall-Conv1D_5_256/StatefulPartitionedCall:output:0batch_norm_5_256_1294349729batch_norm_5_256_1294349731batch_norm_5_256_1294349733batch_norm_5_256_1294349735*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_Batch_Norm_5_256_layer_call_and_return_conditional_losses_12943485072*
(Batch_Norm_5_256/StatefulPartitionedCall╛
(Batch_Norm_3_256/StatefulPartitionedCallStatefulPartitionedCall-Conv1D_3_256/StatefulPartitionedCall:output:0batch_norm_3_256_1294349738batch_norm_3_256_1294349740batch_norm_3_256_1294349742batch_norm_3_256_1294349744*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_Batch_Norm_3_256_layer_call_and_return_conditional_losses_12943483452*
(Batch_Norm_3_256/StatefulPartitionedCallН
Max_Pool_3/PartitionedCallPartitionedCall1Batch_Norm_3_256/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_Max_Pool_3_layer_call_and_return_conditional_losses_12943488142
Max_Pool_3/PartitionedCallН
Max_Pool_5/PartitionedCallPartitionedCall1Batch_Norm_5_256/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_Max_Pool_5_layer_call_and_return_conditional_losses_12943488272
Max_Pool_5/PartitionedCallН
Max_Pool_7/PartitionedCallPartitionedCall1Batch_Norm_7_256/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_Max_Pool_7_layer_call_and_return_conditional_losses_12943488402
Max_Pool_7/PartitionedCall▌
 Pool_Concatenate/PartitionedCallPartitionedCall#Max_Pool_3/PartitionedCall:output:0#Max_Pool_5/PartitionedCall:output:0#Max_Pool_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_Pool_Concatenate_layer_call_and_return_conditional_losses_12943490872"
 Pool_Concatenate/PartitionedCallа
#Concatenate_Dropout/PartitionedCallPartitionedCall)Pool_Concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_Concatenate_Dropout_layer_call_and_return_conditional_losses_12943490942%
#Concatenate_Dropout/PartitionedCall╙
"Prediction/StatefulPartitionedCallStatefulPartitionedCall,Concatenate_Dropout/PartitionedCall:output:0prediction_1294349752prediction_1294349754*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_Prediction_layer_call_and_return_conditional_losses_12943491072$
"Prediction/StatefulPartitionedCallР
IdentityIdentity+Prediction/StatefulPartitionedCall:output:0)^Batch_Norm_3_128/StatefulPartitionedCall)^Batch_Norm_3_256/StatefulPartitionedCall)^Batch_Norm_5_128/StatefulPartitionedCall)^Batch_Norm_5_256/StatefulPartitionedCall)^Batch_Norm_7_128/StatefulPartitionedCall)^Batch_Norm_7_256/StatefulPartitionedCall%^Conv1D_3_128/StatefulPartitionedCall%^Conv1D_3_256/StatefulPartitionedCall%^Conv1D_5_128/StatefulPartitionedCall%^Conv1D_5_256/StatefulPartitionedCall%^Conv1D_7_128/StatefulPartitionedCall%^Conv1D_7_256/StatefulPartitionedCall#^Prediction/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*А
_input_shapeso
m:                  А: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2T
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
!:                  А
 
_user_specified_nameInputs
№*
э
P__inference_Batch_Norm_5_256_layer_call_and_return_conditional_losses_1294348567

inputs6
'assignmovingavg_readvariableop_resource:	А8
)assignmovingavg_1_readvariableop_resource:	А4
%batchnorm_mul_readvariableop_resource:	А0
!batchnorm_readvariableop_resource:	А
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesФ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/meanБ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:А2
moments/StopGradient▓
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:                  А2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╖
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/varianceВ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/SqueezeК
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayе
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOpЩ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:А2
AssignMovingAvg/subР
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:А2
AssignMovingAvg/mul┐
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
╫#<2
AssignMovingAvg_1/decayл
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOpб
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:А2
AssignMovingAvg_1/subШ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:А2
AssignMovingAvg_1/mul╔
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
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/add_1Щ
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  А: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
№
Ш	
1__inference_Tree_Encoder_layer_call_fn_1294349193

inputs
unknown:АА
	unknown_0:	А!
	unknown_1:АА
	unknown_2:	А!
	unknown_3:АА
	unknown_4:	А
	unknown_5:	А
	unknown_6:	А
	unknown_7:	А
	unknown_8:	А
	unknown_9:	А

unknown_10:	А

unknown_11:	А

unknown_12:	А

unknown_13:	А

unknown_14:	А

unknown_15:	А

unknown_16:	А"

unknown_17:АА

unknown_18:	А"

unknown_19:АА

unknown_20:	А"

unknown_21:АА

unknown_22:	А

unknown_23:	А

unknown_24:	А

unknown_25:	А

unknown_26:	А

unknown_27:	А

unknown_28:	А

unknown_29:	А

unknown_30:	А

unknown_31:	А

unknown_32:	А

unknown_33:	А

unknown_34:	А

unknown_35:	А

unknown_36:
identityИвStatefulPartitionedCallю
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
:         *H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_Tree_Encoder_layer_call_and_return_conditional_losses_12943491142
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*А
_input_shapeso
m:                  А: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameInputs
ё
│
P__inference_Batch_Norm_3_128_layer_call_and_return_conditional_losses_1294350801

inputs0
!batchnorm_readvariableop_resource:	А4
%batchnorm_mul_readvariableop_resource:	А2
#batchnorm_readvariableop_1_resource:	А2
#batchnorm_readvariableop_2_resource:	А
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/add_1щ
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  А: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
Ў
К
P__inference_Pool_Concatenate_layer_call_and_return_conditional_losses_1294351340
inputs_0
inputs_1
inputs_2
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisМ
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*(
_output_shapes
:         А2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:         А:         А:         А:R N
(
_output_shapes
:         А
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:         А
"
_user_specified_name
inputs/1:RN
(
_output_shapes
:         А
"
_user_specified_name
inputs/2
┬
r
S__inference_Concatenate_Dropout_layer_call_and_return_conditional_losses_1294349223

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         А2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╡
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
э
е
1__inference_Conv1D_7_128_layer_call_fn_1294350734

inputs
unknown:АА
	unknown_0:	А
identityИвStatefulPartitionedCallН
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_Conv1D_7_128_layer_call_and_return_conditional_losses_12943488812
StatefulPartitionedCallЬ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:                  А: : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
№*
э
P__inference_Batch_Norm_3_128_layer_call_and_return_conditional_losses_1294347919

inputs6
'assignmovingavg_readvariableop_resource:	А8
)assignmovingavg_1_readvariableop_resource:	А4
%batchnorm_mul_readvariableop_resource:	А0
!batchnorm_readvariableop_resource:	А
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesФ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/meanБ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:А2
moments/StopGradient▓
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:                  А2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╖
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/varianceВ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/SqueezeК
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayе
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOpЩ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:А2
AssignMovingAvg/subР
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:А2
AssignMovingAvg/mul┐
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
╫#<2
AssignMovingAvg_1/decayл
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOpб
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:А2
AssignMovingAvg_1/subШ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:А2
AssignMovingAvg_1/mul╔
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
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/add_1Щ
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  А: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
С
R
6__inference_Embedding_Dropout_layer_call_fn_1294350643

inputs
identityр
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_Embedding_Dropout_layer_call_and_return_conditional_losses_12943488582
PartitionedCallz
IdentityIdentityPartitionedCall:output:0*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:                  А:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
э
е
1__inference_Conv1D_7_256_layer_call_fn_1294351064

inputs
unknown:АА
	unknown_0:	А
identityИвStatefulPartitionedCallН
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_Conv1D_7_256_layer_call_and_return_conditional_losses_12943489892
StatefulPartitionedCallЬ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:                  А: : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
№*
э
P__inference_Batch_Norm_3_128_layer_call_and_return_conditional_losses_1294350835

inputs6
'assignmovingavg_readvariableop_resource:	А8
)assignmovingavg_1_readvariableop_resource:	А4
%batchnorm_mul_readvariableop_resource:	А0
!batchnorm_readvariableop_resource:	А
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesФ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/meanБ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:А2
moments/StopGradient▓
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:                  А2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╖
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/varianceВ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/SqueezeК
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayе
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOpЩ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:А2
AssignMovingAvg/subР
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:А2
AssignMovingAvg/mul┐
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
╫#<2
AssignMovingAvg_1/decayл
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOpб
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:А2
AssignMovingAvg_1/subШ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:А2
AssignMovingAvg_1/mul╔
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
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/add_1Щ
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  А: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
№
а
L__inference_Conv1D_5_128_layer_call_and_return_conditional_losses_1294348908

inputsC
+conv1d_expanddims_1_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А

identity_1ИвBiasAdd/ReadVariableOpв"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimа
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#                  А2
conv1d/ExpandDims║
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:АА*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim╣
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:АА2
conv1d/ExpandDims_1└
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#                  А*
paddingSAME*
strides
2
conv1dЬ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*5
_output_shapes#
!:                  А*
squeeze_dims

¤        2
conv1d/SqueezeН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЦ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:                  А2	
BiasAddo
SigmoidSigmoidBiasAdd:output:0*
T0*5
_output_shapes#
!:                  А2	
Sigmoidp
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*5
_output_shapes#
!:                  А2
muli
IdentityIdentitymul:z:0*
T0*5
_output_shapes#
!:                  А2

Identity╒
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1294348901*V
_output_shapesD
B:                  А:                  А2
	IdentityN╢

Identity_1IdentityIdentityN:output:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*5
_output_shapes#
!:                  А2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:                  А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
й
f
J__inference_Max_Pool_7_layer_call_and_return_conditional_losses_1294348840

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
:                  2
Maxi
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:                  2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
й
f
J__inference_Max_Pool_5_layer_call_and_return_conditional_losses_1294348827

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
:                  2
Maxi
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:                  2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
№
а
L__inference_Conv1D_3_256_layer_call_and_return_conditional_losses_1294349043

inputsC
+conv1d_expanddims_1_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А

identity_1ИвBiasAdd/ReadVariableOpв"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimа
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#                  А2
conv1d/ExpandDims║
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:АА*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim╣
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:АА2
conv1d/ExpandDims_1└
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#                  А*
paddingSAME*
strides
2
conv1dЬ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*5
_output_shapes#
!:                  А*
squeeze_dims

¤        2
conv1d/SqueezeН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЦ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:                  А2	
BiasAddo
SigmoidSigmoidBiasAdd:output:0*
T0*5
_output_shapes#
!:                  А2	
Sigmoidp
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*5
_output_shapes#
!:                  А2
muli
IdentityIdentitymul:z:0*
T0*5
_output_shapes#
!:                  А2

Identity╒
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1294349036*V
_output_shapesD
B:                  А:                  А2
	IdentityN╢

Identity_1IdentityIdentityN:output:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*5
_output_shapes#
!:                  А2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:                  А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
╢
o
Q__inference_Embedding_Dropout_layer_call_and_return_conditional_losses_1294350653

inputs

identity_1h
IdentityIdentityinputs*
T0*5
_output_shapes#
!:                  А2

Identityw

Identity_1IdentityIdentity:output:0*
T0*5
_output_shapes#
!:                  А2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:                  А:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
Й
K
/__inference_Max_Pool_5_layer_call_fn_1294348833

inputs
identity╘
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:                  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_Max_Pool_5_layer_call_and_return_conditional_losses_12943488272
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:                  2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
ё
│
P__inference_Batch_Norm_5_256_layer_call_and_return_conditional_losses_1294348507

inputs0
!batchnorm_readvariableop_resource:	А4
%batchnorm_mul_readvariableop_resource:	А2
#batchnorm_readvariableop_1_resource:	А2
#batchnorm_readvariableop_2_resource:	А
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/add_1щ
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  А: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
╛
╘
5__inference_Batch_Norm_7_256_layer_call_fn_1294351258

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_Batch_Norm_7_256_layer_call_and_return_conditional_losses_12943486692
StatefulPartitionedCallЬ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
ё
│
P__inference_Batch_Norm_3_256_layer_call_and_return_conditional_losses_1294351131

inputs0
!batchnorm_readvariableop_resource:	А4
%batchnorm_mul_readvariableop_resource:	А2
#batchnorm_readvariableop_1_resource:	А2
#batchnorm_readvariableop_2_resource:	А
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/add_1щ
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  А: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
юП
Д'
L__inference_Tree_Encoder_layer_call_and_return_conditional_losses_1294350638

inputsP
8conv1d_7_128_conv1d_expanddims_1_readvariableop_resource:АА;
,conv1d_7_128_biasadd_readvariableop_resource:	АP
8conv1d_5_128_conv1d_expanddims_1_readvariableop_resource:АА;
,conv1d_5_128_biasadd_readvariableop_resource:	АP
8conv1d_3_128_conv1d_expanddims_1_readvariableop_resource:АА;
,conv1d_3_128_biasadd_readvariableop_resource:	АG
8batch_norm_7_128_assignmovingavg_readvariableop_resource:	АI
:batch_norm_7_128_assignmovingavg_1_readvariableop_resource:	АE
6batch_norm_7_128_batchnorm_mul_readvariableop_resource:	АA
2batch_norm_7_128_batchnorm_readvariableop_resource:	АG
8batch_norm_5_128_assignmovingavg_readvariableop_resource:	АI
:batch_norm_5_128_assignmovingavg_1_readvariableop_resource:	АE
6batch_norm_5_128_batchnorm_mul_readvariableop_resource:	АA
2batch_norm_5_128_batchnorm_readvariableop_resource:	АG
8batch_norm_3_128_assignmovingavg_readvariableop_resource:	АI
:batch_norm_3_128_assignmovingavg_1_readvariableop_resource:	АE
6batch_norm_3_128_batchnorm_mul_readvariableop_resource:	АA
2batch_norm_3_128_batchnorm_readvariableop_resource:	АP
8conv1d_7_256_conv1d_expanddims_1_readvariableop_resource:АА;
,conv1d_7_256_biasadd_readvariableop_resource:	АP
8conv1d_5_256_conv1d_expanddims_1_readvariableop_resource:АА;
,conv1d_5_256_biasadd_readvariableop_resource:	АP
8conv1d_3_256_conv1d_expanddims_1_readvariableop_resource:АА;
,conv1d_3_256_biasadd_readvariableop_resource:	АG
8batch_norm_7_256_assignmovingavg_readvariableop_resource:	АI
:batch_norm_7_256_assignmovingavg_1_readvariableop_resource:	АE
6batch_norm_7_256_batchnorm_mul_readvariableop_resource:	АA
2batch_norm_7_256_batchnorm_readvariableop_resource:	АG
8batch_norm_5_256_assignmovingavg_readvariableop_resource:	АI
:batch_norm_5_256_assignmovingavg_1_readvariableop_resource:	АE
6batch_norm_5_256_batchnorm_mul_readvariableop_resource:	АA
2batch_norm_5_256_batchnorm_readvariableop_resource:	АG
8batch_norm_3_256_assignmovingavg_readvariableop_resource:	АI
:batch_norm_3_256_assignmovingavg_1_readvariableop_resource:	АE
6batch_norm_3_256_batchnorm_mul_readvariableop_resource:	АA
2batch_norm_3_256_batchnorm_readvariableop_resource:	А<
)prediction_matmul_readvariableop_resource:	А8
*prediction_biasadd_readvariableop_resource:
identityИв Batch_Norm_3_128/AssignMovingAvgв/Batch_Norm_3_128/AssignMovingAvg/ReadVariableOpв"Batch_Norm_3_128/AssignMovingAvg_1в1Batch_Norm_3_128/AssignMovingAvg_1/ReadVariableOpв)Batch_Norm_3_128/batchnorm/ReadVariableOpв-Batch_Norm_3_128/batchnorm/mul/ReadVariableOpв Batch_Norm_3_256/AssignMovingAvgв/Batch_Norm_3_256/AssignMovingAvg/ReadVariableOpв"Batch_Norm_3_256/AssignMovingAvg_1в1Batch_Norm_3_256/AssignMovingAvg_1/ReadVariableOpв)Batch_Norm_3_256/batchnorm/ReadVariableOpв-Batch_Norm_3_256/batchnorm/mul/ReadVariableOpв Batch_Norm_5_128/AssignMovingAvgв/Batch_Norm_5_128/AssignMovingAvg/ReadVariableOpв"Batch_Norm_5_128/AssignMovingAvg_1в1Batch_Norm_5_128/AssignMovingAvg_1/ReadVariableOpв)Batch_Norm_5_128/batchnorm/ReadVariableOpв-Batch_Norm_5_128/batchnorm/mul/ReadVariableOpв Batch_Norm_5_256/AssignMovingAvgв/Batch_Norm_5_256/AssignMovingAvg/ReadVariableOpв"Batch_Norm_5_256/AssignMovingAvg_1в1Batch_Norm_5_256/AssignMovingAvg_1/ReadVariableOpв)Batch_Norm_5_256/batchnorm/ReadVariableOpв-Batch_Norm_5_256/batchnorm/mul/ReadVariableOpв Batch_Norm_7_128/AssignMovingAvgв/Batch_Norm_7_128/AssignMovingAvg/ReadVariableOpв"Batch_Norm_7_128/AssignMovingAvg_1в1Batch_Norm_7_128/AssignMovingAvg_1/ReadVariableOpв)Batch_Norm_7_128/batchnorm/ReadVariableOpв-Batch_Norm_7_128/batchnorm/mul/ReadVariableOpв Batch_Norm_7_256/AssignMovingAvgв/Batch_Norm_7_256/AssignMovingAvg/ReadVariableOpв"Batch_Norm_7_256/AssignMovingAvg_1в1Batch_Norm_7_256/AssignMovingAvg_1/ReadVariableOpв)Batch_Norm_7_256/batchnorm/ReadVariableOpв-Batch_Norm_7_256/batchnorm/mul/ReadVariableOpв#Conv1D_3_128/BiasAdd/ReadVariableOpв/Conv1D_3_128/conv1d/ExpandDims_1/ReadVariableOpв#Conv1D_3_256/BiasAdd/ReadVariableOpв/Conv1D_3_256/conv1d/ExpandDims_1/ReadVariableOpв#Conv1D_5_128/BiasAdd/ReadVariableOpв/Conv1D_5_128/conv1d/ExpandDims_1/ReadVariableOpв#Conv1D_5_256/BiasAdd/ReadVariableOpв/Conv1D_5_256/conv1d/ExpandDims_1/ReadVariableOpв#Conv1D_7_128/BiasAdd/ReadVariableOpв/Conv1D_7_128/conv1d/ExpandDims_1/ReadVariableOpв#Conv1D_7_256/BiasAdd/ReadVariableOpв/Conv1D_7_256/conv1d/ExpandDims_1/ReadVariableOpв!Prediction/BiasAdd/ReadVariableOpв Prediction/MatMul/ReadVariableOpЗ
Embedding_Dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?2!
Embedding_Dropout/dropout/Const╖
Embedding_Dropout/dropout/MulMulinputs(Embedding_Dropout/dropout/Const:output:0*
T0*5
_output_shapes#
!:                  А2
Embedding_Dropout/dropout/Mulx
Embedding_Dropout/dropout/ShapeShapeinputs*
T0*
_output_shapes
:2!
Embedding_Dropout/dropout/Shape°
6Embedding_Dropout/dropout/random_uniform/RandomUniformRandomUniform(Embedding_Dropout/dropout/Shape:output:0*
T0*5
_output_shapes#
!:                  А*
dtype028
6Embedding_Dropout/dropout/random_uniform/RandomUniformЩ
(Embedding_Dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2*
(Embedding_Dropout/dropout/GreaterEqual/yФ
&Embedding_Dropout/dropout/GreaterEqualGreaterEqual?Embedding_Dropout/dropout/random_uniform/RandomUniform:output:01Embedding_Dropout/dropout/GreaterEqual/y:output:0*
T0*5
_output_shapes#
!:                  А2(
&Embedding_Dropout/dropout/GreaterEqual├
Embedding_Dropout/dropout/CastCast*Embedding_Dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*5
_output_shapes#
!:                  А2 
Embedding_Dropout/dropout/Cast╨
Embedding_Dropout/dropout/Mul_1Mul!Embedding_Dropout/dropout/Mul:z:0"Embedding_Dropout/dropout/Cast:y:0*
T0*5
_output_shapes#
!:                  А2!
Embedding_Dropout/dropout/Mul_1У
"Conv1D_7_128/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2$
"Conv1D_7_128/conv1d/ExpandDims/dimф
Conv1D_7_128/conv1d/ExpandDims
ExpandDims#Embedding_Dropout/dropout/Mul_1:z:0+Conv1D_7_128/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#                  А2 
Conv1D_7_128/conv1d/ExpandDimsс
/Conv1D_7_128/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp8conv1d_7_128_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:АА*
dtype021
/Conv1D_7_128/conv1d/ExpandDims_1/ReadVariableOpО
$Conv1D_7_128/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Conv1D_7_128/conv1d/ExpandDims_1/dimэ
 Conv1D_7_128/conv1d/ExpandDims_1
ExpandDims7Conv1D_7_128/conv1d/ExpandDims_1/ReadVariableOp:value:0-Conv1D_7_128/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:АА2"
 Conv1D_7_128/conv1d/ExpandDims_1Ї
Conv1D_7_128/conv1dConv2D'Conv1D_7_128/conv1d/ExpandDims:output:0)Conv1D_7_128/conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#                  А*
paddingSAME*
strides
2
Conv1D_7_128/conv1d├
Conv1D_7_128/conv1d/SqueezeSqueezeConv1D_7_128/conv1d:output:0*
T0*5
_output_shapes#
!:                  А*
squeeze_dims

¤        2
Conv1D_7_128/conv1d/Squeeze┤
#Conv1D_7_128/BiasAdd/ReadVariableOpReadVariableOp,conv1d_7_128_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#Conv1D_7_128/BiasAdd/ReadVariableOp╩
Conv1D_7_128/BiasAddBiasAdd$Conv1D_7_128/conv1d/Squeeze:output:0+Conv1D_7_128/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:                  А2
Conv1D_7_128/BiasAddЦ
Conv1D_7_128/SigmoidSigmoidConv1D_7_128/BiasAdd:output:0*
T0*5
_output_shapes#
!:                  А2
Conv1D_7_128/Sigmoidд
Conv1D_7_128/mulMulConv1D_7_128/BiasAdd:output:0Conv1D_7_128/Sigmoid:y:0*
T0*5
_output_shapes#
!:                  А2
Conv1D_7_128/mulР
Conv1D_7_128/IdentityIdentityConv1D_7_128/mul:z:0*
T0*5
_output_shapes#
!:                  А2
Conv1D_7_128/IdentityЙ
Conv1D_7_128/IdentityN	IdentityNConv1D_7_128/mul:z:0Conv1D_7_128/BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1294350343*V
_output_shapesD
B:                  А:                  А2
Conv1D_7_128/IdentityNУ
"Conv1D_5_128/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2$
"Conv1D_5_128/conv1d/ExpandDims/dimф
Conv1D_5_128/conv1d/ExpandDims
ExpandDims#Embedding_Dropout/dropout/Mul_1:z:0+Conv1D_5_128/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#                  А2 
Conv1D_5_128/conv1d/ExpandDimsс
/Conv1D_5_128/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp8conv1d_5_128_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:АА*
dtype021
/Conv1D_5_128/conv1d/ExpandDims_1/ReadVariableOpО
$Conv1D_5_128/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Conv1D_5_128/conv1d/ExpandDims_1/dimэ
 Conv1D_5_128/conv1d/ExpandDims_1
ExpandDims7Conv1D_5_128/conv1d/ExpandDims_1/ReadVariableOp:value:0-Conv1D_5_128/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:АА2"
 Conv1D_5_128/conv1d/ExpandDims_1Ї
Conv1D_5_128/conv1dConv2D'Conv1D_5_128/conv1d/ExpandDims:output:0)Conv1D_5_128/conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#                  А*
paddingSAME*
strides
2
Conv1D_5_128/conv1d├
Conv1D_5_128/conv1d/SqueezeSqueezeConv1D_5_128/conv1d:output:0*
T0*5
_output_shapes#
!:                  А*
squeeze_dims

¤        2
Conv1D_5_128/conv1d/Squeeze┤
#Conv1D_5_128/BiasAdd/ReadVariableOpReadVariableOp,conv1d_5_128_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#Conv1D_5_128/BiasAdd/ReadVariableOp╩
Conv1D_5_128/BiasAddBiasAdd$Conv1D_5_128/conv1d/Squeeze:output:0+Conv1D_5_128/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:                  А2
Conv1D_5_128/BiasAddЦ
Conv1D_5_128/SigmoidSigmoidConv1D_5_128/BiasAdd:output:0*
T0*5
_output_shapes#
!:                  А2
Conv1D_5_128/Sigmoidд
Conv1D_5_128/mulMulConv1D_5_128/BiasAdd:output:0Conv1D_5_128/Sigmoid:y:0*
T0*5
_output_shapes#
!:                  А2
Conv1D_5_128/mulР
Conv1D_5_128/IdentityIdentityConv1D_5_128/mul:z:0*
T0*5
_output_shapes#
!:                  А2
Conv1D_5_128/IdentityЙ
Conv1D_5_128/IdentityN	IdentityNConv1D_5_128/mul:z:0Conv1D_5_128/BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1294350360*V
_output_shapesD
B:                  А:                  А2
Conv1D_5_128/IdentityNУ
"Conv1D_3_128/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2$
"Conv1D_3_128/conv1d/ExpandDims/dimф
Conv1D_3_128/conv1d/ExpandDims
ExpandDims#Embedding_Dropout/dropout/Mul_1:z:0+Conv1D_3_128/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#                  А2 
Conv1D_3_128/conv1d/ExpandDimsс
/Conv1D_3_128/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp8conv1d_3_128_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:АА*
dtype021
/Conv1D_3_128/conv1d/ExpandDims_1/ReadVariableOpО
$Conv1D_3_128/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Conv1D_3_128/conv1d/ExpandDims_1/dimэ
 Conv1D_3_128/conv1d/ExpandDims_1
ExpandDims7Conv1D_3_128/conv1d/ExpandDims_1/ReadVariableOp:value:0-Conv1D_3_128/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:АА2"
 Conv1D_3_128/conv1d/ExpandDims_1Ї
Conv1D_3_128/conv1dConv2D'Conv1D_3_128/conv1d/ExpandDims:output:0)Conv1D_3_128/conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#                  А*
paddingSAME*
strides
2
Conv1D_3_128/conv1d├
Conv1D_3_128/conv1d/SqueezeSqueezeConv1D_3_128/conv1d:output:0*
T0*5
_output_shapes#
!:                  А*
squeeze_dims

¤        2
Conv1D_3_128/conv1d/Squeeze┤
#Conv1D_3_128/BiasAdd/ReadVariableOpReadVariableOp,conv1d_3_128_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#Conv1D_3_128/BiasAdd/ReadVariableOp╩
Conv1D_3_128/BiasAddBiasAdd$Conv1D_3_128/conv1d/Squeeze:output:0+Conv1D_3_128/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:                  А2
Conv1D_3_128/BiasAddЦ
Conv1D_3_128/SigmoidSigmoidConv1D_3_128/BiasAdd:output:0*
T0*5
_output_shapes#
!:                  А2
Conv1D_3_128/Sigmoidд
Conv1D_3_128/mulMulConv1D_3_128/BiasAdd:output:0Conv1D_3_128/Sigmoid:y:0*
T0*5
_output_shapes#
!:                  А2
Conv1D_3_128/mulР
Conv1D_3_128/IdentityIdentityConv1D_3_128/mul:z:0*
T0*5
_output_shapes#
!:                  А2
Conv1D_3_128/IdentityЙ
Conv1D_3_128/IdentityN	IdentityNConv1D_3_128/mul:z:0Conv1D_3_128/BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1294350377*V
_output_shapesD
B:                  А:                  А2
Conv1D_3_128/IdentityN│
/Batch_Norm_7_128/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       21
/Batch_Norm_7_128/moments/mean/reduction_indicesр
Batch_Norm_7_128/moments/meanMeanConv1D_7_128/IdentityN:output:08Batch_Norm_7_128/moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
Batch_Norm_7_128/moments/mean┤
%Batch_Norm_7_128/moments/StopGradientStopGradient&Batch_Norm_7_128/moments/mean:output:0*
T0*#
_output_shapes
:А2'
%Batch_Norm_7_128/moments/StopGradient■
*Batch_Norm_7_128/moments/SquaredDifferenceSquaredDifferenceConv1D_7_128/IdentityN:output:0.Batch_Norm_7_128/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:                  А2,
*Batch_Norm_7_128/moments/SquaredDifference╗
3Batch_Norm_7_128/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       25
3Batch_Norm_7_128/moments/variance/reduction_indices√
!Batch_Norm_7_128/moments/varianceMean.Batch_Norm_7_128/moments/SquaredDifference:z:0<Batch_Norm_7_128/moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2#
!Batch_Norm_7_128/moments/variance╡
 Batch_Norm_7_128/moments/SqueezeSqueeze&Batch_Norm_7_128/moments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2"
 Batch_Norm_7_128/moments/Squeeze╜
"Batch_Norm_7_128/moments/Squeeze_1Squeeze*Batch_Norm_7_128/moments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2$
"Batch_Norm_7_128/moments/Squeeze_1Х
&Batch_Norm_7_128/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2(
&Batch_Norm_7_128/AssignMovingAvg/decay╪
/Batch_Norm_7_128/AssignMovingAvg/ReadVariableOpReadVariableOp8batch_norm_7_128_assignmovingavg_readvariableop_resource*
_output_shapes	
:А*
dtype021
/Batch_Norm_7_128/AssignMovingAvg/ReadVariableOp▌
$Batch_Norm_7_128/AssignMovingAvg/subSub7Batch_Norm_7_128/AssignMovingAvg/ReadVariableOp:value:0)Batch_Norm_7_128/moments/Squeeze:output:0*
T0*
_output_shapes	
:А2&
$Batch_Norm_7_128/AssignMovingAvg/sub╘
$Batch_Norm_7_128/AssignMovingAvg/mulMul(Batch_Norm_7_128/AssignMovingAvg/sub:z:0/Batch_Norm_7_128/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:А2&
$Batch_Norm_7_128/AssignMovingAvg/mulФ
 Batch_Norm_7_128/AssignMovingAvgAssignSubVariableOp8batch_norm_7_128_assignmovingavg_readvariableop_resource(Batch_Norm_7_128/AssignMovingAvg/mul:z:00^Batch_Norm_7_128/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02"
 Batch_Norm_7_128/AssignMovingAvgЩ
(Batch_Norm_7_128/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2*
(Batch_Norm_7_128/AssignMovingAvg_1/decay▐
1Batch_Norm_7_128/AssignMovingAvg_1/ReadVariableOpReadVariableOp:batch_norm_7_128_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:А*
dtype023
1Batch_Norm_7_128/AssignMovingAvg_1/ReadVariableOpх
&Batch_Norm_7_128/AssignMovingAvg_1/subSub9Batch_Norm_7_128/AssignMovingAvg_1/ReadVariableOp:value:0+Batch_Norm_7_128/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:А2(
&Batch_Norm_7_128/AssignMovingAvg_1/sub▄
&Batch_Norm_7_128/AssignMovingAvg_1/mulMul*Batch_Norm_7_128/AssignMovingAvg_1/sub:z:01Batch_Norm_7_128/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:А2(
&Batch_Norm_7_128/AssignMovingAvg_1/mulЮ
"Batch_Norm_7_128/AssignMovingAvg_1AssignSubVariableOp:batch_norm_7_128_assignmovingavg_1_readvariableop_resource*Batch_Norm_7_128/AssignMovingAvg_1/mul:z:02^Batch_Norm_7_128/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02$
"Batch_Norm_7_128/AssignMovingAvg_1Й
 Batch_Norm_7_128/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2"
 Batch_Norm_7_128/batchnorm/add/y╟
Batch_Norm_7_128/batchnorm/addAddV2+Batch_Norm_7_128/moments/Squeeze_1:output:0)Batch_Norm_7_128/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2 
Batch_Norm_7_128/batchnorm/addЧ
 Batch_Norm_7_128/batchnorm/RsqrtRsqrt"Batch_Norm_7_128/batchnorm/add:z:0*
T0*
_output_shapes	
:А2"
 Batch_Norm_7_128/batchnorm/Rsqrt╥
-Batch_Norm_7_128/batchnorm/mul/ReadVariableOpReadVariableOp6batch_norm_7_128_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-Batch_Norm_7_128/batchnorm/mul/ReadVariableOp╩
Batch_Norm_7_128/batchnorm/mulMul$Batch_Norm_7_128/batchnorm/Rsqrt:y:05Batch_Norm_7_128/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2 
Batch_Norm_7_128/batchnorm/mul╨
 Batch_Norm_7_128/batchnorm/mul_1MulConv1D_7_128/IdentityN:output:0"Batch_Norm_7_128/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2"
 Batch_Norm_7_128/batchnorm/mul_1└
 Batch_Norm_7_128/batchnorm/mul_2Mul)Batch_Norm_7_128/moments/Squeeze:output:0"Batch_Norm_7_128/batchnorm/mul:z:0*
T0*
_output_shapes	
:А2"
 Batch_Norm_7_128/batchnorm/mul_2╞
)Batch_Norm_7_128/batchnorm/ReadVariableOpReadVariableOp2batch_norm_7_128_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)Batch_Norm_7_128/batchnorm/ReadVariableOp╞
Batch_Norm_7_128/batchnorm/subSub1Batch_Norm_7_128/batchnorm/ReadVariableOp:value:0$Batch_Norm_7_128/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2 
Batch_Norm_7_128/batchnorm/sub╫
 Batch_Norm_7_128/batchnorm/add_1AddV2$Batch_Norm_7_128/batchnorm/mul_1:z:0"Batch_Norm_7_128/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2"
 Batch_Norm_7_128/batchnorm/add_1│
/Batch_Norm_5_128/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       21
/Batch_Norm_5_128/moments/mean/reduction_indicesр
Batch_Norm_5_128/moments/meanMeanConv1D_5_128/IdentityN:output:08Batch_Norm_5_128/moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
Batch_Norm_5_128/moments/mean┤
%Batch_Norm_5_128/moments/StopGradientStopGradient&Batch_Norm_5_128/moments/mean:output:0*
T0*#
_output_shapes
:А2'
%Batch_Norm_5_128/moments/StopGradient■
*Batch_Norm_5_128/moments/SquaredDifferenceSquaredDifferenceConv1D_5_128/IdentityN:output:0.Batch_Norm_5_128/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:                  А2,
*Batch_Norm_5_128/moments/SquaredDifference╗
3Batch_Norm_5_128/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       25
3Batch_Norm_5_128/moments/variance/reduction_indices√
!Batch_Norm_5_128/moments/varianceMean.Batch_Norm_5_128/moments/SquaredDifference:z:0<Batch_Norm_5_128/moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2#
!Batch_Norm_5_128/moments/variance╡
 Batch_Norm_5_128/moments/SqueezeSqueeze&Batch_Norm_5_128/moments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2"
 Batch_Norm_5_128/moments/Squeeze╜
"Batch_Norm_5_128/moments/Squeeze_1Squeeze*Batch_Norm_5_128/moments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2$
"Batch_Norm_5_128/moments/Squeeze_1Х
&Batch_Norm_5_128/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2(
&Batch_Norm_5_128/AssignMovingAvg/decay╪
/Batch_Norm_5_128/AssignMovingAvg/ReadVariableOpReadVariableOp8batch_norm_5_128_assignmovingavg_readvariableop_resource*
_output_shapes	
:А*
dtype021
/Batch_Norm_5_128/AssignMovingAvg/ReadVariableOp▌
$Batch_Norm_5_128/AssignMovingAvg/subSub7Batch_Norm_5_128/AssignMovingAvg/ReadVariableOp:value:0)Batch_Norm_5_128/moments/Squeeze:output:0*
T0*
_output_shapes	
:А2&
$Batch_Norm_5_128/AssignMovingAvg/sub╘
$Batch_Norm_5_128/AssignMovingAvg/mulMul(Batch_Norm_5_128/AssignMovingAvg/sub:z:0/Batch_Norm_5_128/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:А2&
$Batch_Norm_5_128/AssignMovingAvg/mulФ
 Batch_Norm_5_128/AssignMovingAvgAssignSubVariableOp8batch_norm_5_128_assignmovingavg_readvariableop_resource(Batch_Norm_5_128/AssignMovingAvg/mul:z:00^Batch_Norm_5_128/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02"
 Batch_Norm_5_128/AssignMovingAvgЩ
(Batch_Norm_5_128/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2*
(Batch_Norm_5_128/AssignMovingAvg_1/decay▐
1Batch_Norm_5_128/AssignMovingAvg_1/ReadVariableOpReadVariableOp:batch_norm_5_128_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:А*
dtype023
1Batch_Norm_5_128/AssignMovingAvg_1/ReadVariableOpх
&Batch_Norm_5_128/AssignMovingAvg_1/subSub9Batch_Norm_5_128/AssignMovingAvg_1/ReadVariableOp:value:0+Batch_Norm_5_128/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:А2(
&Batch_Norm_5_128/AssignMovingAvg_1/sub▄
&Batch_Norm_5_128/AssignMovingAvg_1/mulMul*Batch_Norm_5_128/AssignMovingAvg_1/sub:z:01Batch_Norm_5_128/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:А2(
&Batch_Norm_5_128/AssignMovingAvg_1/mulЮ
"Batch_Norm_5_128/AssignMovingAvg_1AssignSubVariableOp:batch_norm_5_128_assignmovingavg_1_readvariableop_resource*Batch_Norm_5_128/AssignMovingAvg_1/mul:z:02^Batch_Norm_5_128/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02$
"Batch_Norm_5_128/AssignMovingAvg_1Й
 Batch_Norm_5_128/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2"
 Batch_Norm_5_128/batchnorm/add/y╟
Batch_Norm_5_128/batchnorm/addAddV2+Batch_Norm_5_128/moments/Squeeze_1:output:0)Batch_Norm_5_128/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2 
Batch_Norm_5_128/batchnorm/addЧ
 Batch_Norm_5_128/batchnorm/RsqrtRsqrt"Batch_Norm_5_128/batchnorm/add:z:0*
T0*
_output_shapes	
:А2"
 Batch_Norm_5_128/batchnorm/Rsqrt╥
-Batch_Norm_5_128/batchnorm/mul/ReadVariableOpReadVariableOp6batch_norm_5_128_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-Batch_Norm_5_128/batchnorm/mul/ReadVariableOp╩
Batch_Norm_5_128/batchnorm/mulMul$Batch_Norm_5_128/batchnorm/Rsqrt:y:05Batch_Norm_5_128/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2 
Batch_Norm_5_128/batchnorm/mul╨
 Batch_Norm_5_128/batchnorm/mul_1MulConv1D_5_128/IdentityN:output:0"Batch_Norm_5_128/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2"
 Batch_Norm_5_128/batchnorm/mul_1└
 Batch_Norm_5_128/batchnorm/mul_2Mul)Batch_Norm_5_128/moments/Squeeze:output:0"Batch_Norm_5_128/batchnorm/mul:z:0*
T0*
_output_shapes	
:А2"
 Batch_Norm_5_128/batchnorm/mul_2╞
)Batch_Norm_5_128/batchnorm/ReadVariableOpReadVariableOp2batch_norm_5_128_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)Batch_Norm_5_128/batchnorm/ReadVariableOp╞
Batch_Norm_5_128/batchnorm/subSub1Batch_Norm_5_128/batchnorm/ReadVariableOp:value:0$Batch_Norm_5_128/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2 
Batch_Norm_5_128/batchnorm/sub╫
 Batch_Norm_5_128/batchnorm/add_1AddV2$Batch_Norm_5_128/batchnorm/mul_1:z:0"Batch_Norm_5_128/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2"
 Batch_Norm_5_128/batchnorm/add_1│
/Batch_Norm_3_128/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       21
/Batch_Norm_3_128/moments/mean/reduction_indicesр
Batch_Norm_3_128/moments/meanMeanConv1D_3_128/IdentityN:output:08Batch_Norm_3_128/moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
Batch_Norm_3_128/moments/mean┤
%Batch_Norm_3_128/moments/StopGradientStopGradient&Batch_Norm_3_128/moments/mean:output:0*
T0*#
_output_shapes
:А2'
%Batch_Norm_3_128/moments/StopGradient■
*Batch_Norm_3_128/moments/SquaredDifferenceSquaredDifferenceConv1D_3_128/IdentityN:output:0.Batch_Norm_3_128/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:                  А2,
*Batch_Norm_3_128/moments/SquaredDifference╗
3Batch_Norm_3_128/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       25
3Batch_Norm_3_128/moments/variance/reduction_indices√
!Batch_Norm_3_128/moments/varianceMean.Batch_Norm_3_128/moments/SquaredDifference:z:0<Batch_Norm_3_128/moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2#
!Batch_Norm_3_128/moments/variance╡
 Batch_Norm_3_128/moments/SqueezeSqueeze&Batch_Norm_3_128/moments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2"
 Batch_Norm_3_128/moments/Squeeze╜
"Batch_Norm_3_128/moments/Squeeze_1Squeeze*Batch_Norm_3_128/moments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2$
"Batch_Norm_3_128/moments/Squeeze_1Х
&Batch_Norm_3_128/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2(
&Batch_Norm_3_128/AssignMovingAvg/decay╪
/Batch_Norm_3_128/AssignMovingAvg/ReadVariableOpReadVariableOp8batch_norm_3_128_assignmovingavg_readvariableop_resource*
_output_shapes	
:А*
dtype021
/Batch_Norm_3_128/AssignMovingAvg/ReadVariableOp▌
$Batch_Norm_3_128/AssignMovingAvg/subSub7Batch_Norm_3_128/AssignMovingAvg/ReadVariableOp:value:0)Batch_Norm_3_128/moments/Squeeze:output:0*
T0*
_output_shapes	
:А2&
$Batch_Norm_3_128/AssignMovingAvg/sub╘
$Batch_Norm_3_128/AssignMovingAvg/mulMul(Batch_Norm_3_128/AssignMovingAvg/sub:z:0/Batch_Norm_3_128/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:А2&
$Batch_Norm_3_128/AssignMovingAvg/mulФ
 Batch_Norm_3_128/AssignMovingAvgAssignSubVariableOp8batch_norm_3_128_assignmovingavg_readvariableop_resource(Batch_Norm_3_128/AssignMovingAvg/mul:z:00^Batch_Norm_3_128/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02"
 Batch_Norm_3_128/AssignMovingAvgЩ
(Batch_Norm_3_128/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2*
(Batch_Norm_3_128/AssignMovingAvg_1/decay▐
1Batch_Norm_3_128/AssignMovingAvg_1/ReadVariableOpReadVariableOp:batch_norm_3_128_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:А*
dtype023
1Batch_Norm_3_128/AssignMovingAvg_1/ReadVariableOpх
&Batch_Norm_3_128/AssignMovingAvg_1/subSub9Batch_Norm_3_128/AssignMovingAvg_1/ReadVariableOp:value:0+Batch_Norm_3_128/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:А2(
&Batch_Norm_3_128/AssignMovingAvg_1/sub▄
&Batch_Norm_3_128/AssignMovingAvg_1/mulMul*Batch_Norm_3_128/AssignMovingAvg_1/sub:z:01Batch_Norm_3_128/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:А2(
&Batch_Norm_3_128/AssignMovingAvg_1/mulЮ
"Batch_Norm_3_128/AssignMovingAvg_1AssignSubVariableOp:batch_norm_3_128_assignmovingavg_1_readvariableop_resource*Batch_Norm_3_128/AssignMovingAvg_1/mul:z:02^Batch_Norm_3_128/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02$
"Batch_Norm_3_128/AssignMovingAvg_1Й
 Batch_Norm_3_128/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2"
 Batch_Norm_3_128/batchnorm/add/y╟
Batch_Norm_3_128/batchnorm/addAddV2+Batch_Norm_3_128/moments/Squeeze_1:output:0)Batch_Norm_3_128/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2 
Batch_Norm_3_128/batchnorm/addЧ
 Batch_Norm_3_128/batchnorm/RsqrtRsqrt"Batch_Norm_3_128/batchnorm/add:z:0*
T0*
_output_shapes	
:А2"
 Batch_Norm_3_128/batchnorm/Rsqrt╥
-Batch_Norm_3_128/batchnorm/mul/ReadVariableOpReadVariableOp6batch_norm_3_128_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-Batch_Norm_3_128/batchnorm/mul/ReadVariableOp╩
Batch_Norm_3_128/batchnorm/mulMul$Batch_Norm_3_128/batchnorm/Rsqrt:y:05Batch_Norm_3_128/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2 
Batch_Norm_3_128/batchnorm/mul╨
 Batch_Norm_3_128/batchnorm/mul_1MulConv1D_3_128/IdentityN:output:0"Batch_Norm_3_128/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2"
 Batch_Norm_3_128/batchnorm/mul_1└
 Batch_Norm_3_128/batchnorm/mul_2Mul)Batch_Norm_3_128/moments/Squeeze:output:0"Batch_Norm_3_128/batchnorm/mul:z:0*
T0*
_output_shapes	
:А2"
 Batch_Norm_3_128/batchnorm/mul_2╞
)Batch_Norm_3_128/batchnorm/ReadVariableOpReadVariableOp2batch_norm_3_128_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)Batch_Norm_3_128/batchnorm/ReadVariableOp╞
Batch_Norm_3_128/batchnorm/subSub1Batch_Norm_3_128/batchnorm/ReadVariableOp:value:0$Batch_Norm_3_128/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2 
Batch_Norm_3_128/batchnorm/sub╫
 Batch_Norm_3_128/batchnorm/add_1AddV2$Batch_Norm_3_128/batchnorm/mul_1:z:0"Batch_Norm_3_128/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2"
 Batch_Norm_3_128/batchnorm/add_1У
"Conv1D_7_256/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2$
"Conv1D_7_256/conv1d/ExpandDims/dimх
Conv1D_7_256/conv1d/ExpandDims
ExpandDims$Batch_Norm_7_128/batchnorm/add_1:z:0+Conv1D_7_256/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#                  А2 
Conv1D_7_256/conv1d/ExpandDimsс
/Conv1D_7_256/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp8conv1d_7_256_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:АА*
dtype021
/Conv1D_7_256/conv1d/ExpandDims_1/ReadVariableOpО
$Conv1D_7_256/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Conv1D_7_256/conv1d/ExpandDims_1/dimэ
 Conv1D_7_256/conv1d/ExpandDims_1
ExpandDims7Conv1D_7_256/conv1d/ExpandDims_1/ReadVariableOp:value:0-Conv1D_7_256/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:АА2"
 Conv1D_7_256/conv1d/ExpandDims_1Ї
Conv1D_7_256/conv1dConv2D'Conv1D_7_256/conv1d/ExpandDims:output:0)Conv1D_7_256/conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#                  А*
paddingSAME*
strides
2
Conv1D_7_256/conv1d├
Conv1D_7_256/conv1d/SqueezeSqueezeConv1D_7_256/conv1d:output:0*
T0*5
_output_shapes#
!:                  А*
squeeze_dims

¤        2
Conv1D_7_256/conv1d/Squeeze┤
#Conv1D_7_256/BiasAdd/ReadVariableOpReadVariableOp,conv1d_7_256_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#Conv1D_7_256/BiasAdd/ReadVariableOp╩
Conv1D_7_256/BiasAddBiasAdd$Conv1D_7_256/conv1d/Squeeze:output:0+Conv1D_7_256/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:                  А2
Conv1D_7_256/BiasAddЦ
Conv1D_7_256/SigmoidSigmoidConv1D_7_256/BiasAdd:output:0*
T0*5
_output_shapes#
!:                  А2
Conv1D_7_256/Sigmoidд
Conv1D_7_256/mulMulConv1D_7_256/BiasAdd:output:0Conv1D_7_256/Sigmoid:y:0*
T0*5
_output_shapes#
!:                  А2
Conv1D_7_256/mulР
Conv1D_7_256/IdentityIdentityConv1D_7_256/mul:z:0*
T0*5
_output_shapes#
!:                  А2
Conv1D_7_256/IdentityЙ
Conv1D_7_256/IdentityN	IdentityNConv1D_7_256/mul:z:0Conv1D_7_256/BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1294350484*V
_output_shapesD
B:                  А:                  А2
Conv1D_7_256/IdentityNУ
"Conv1D_5_256/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2$
"Conv1D_5_256/conv1d/ExpandDims/dimх
Conv1D_5_256/conv1d/ExpandDims
ExpandDims$Batch_Norm_5_128/batchnorm/add_1:z:0+Conv1D_5_256/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#                  А2 
Conv1D_5_256/conv1d/ExpandDimsс
/Conv1D_5_256/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp8conv1d_5_256_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:АА*
dtype021
/Conv1D_5_256/conv1d/ExpandDims_1/ReadVariableOpО
$Conv1D_5_256/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Conv1D_5_256/conv1d/ExpandDims_1/dimэ
 Conv1D_5_256/conv1d/ExpandDims_1
ExpandDims7Conv1D_5_256/conv1d/ExpandDims_1/ReadVariableOp:value:0-Conv1D_5_256/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:АА2"
 Conv1D_5_256/conv1d/ExpandDims_1Ї
Conv1D_5_256/conv1dConv2D'Conv1D_5_256/conv1d/ExpandDims:output:0)Conv1D_5_256/conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#                  А*
paddingSAME*
strides
2
Conv1D_5_256/conv1d├
Conv1D_5_256/conv1d/SqueezeSqueezeConv1D_5_256/conv1d:output:0*
T0*5
_output_shapes#
!:                  А*
squeeze_dims

¤        2
Conv1D_5_256/conv1d/Squeeze┤
#Conv1D_5_256/BiasAdd/ReadVariableOpReadVariableOp,conv1d_5_256_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#Conv1D_5_256/BiasAdd/ReadVariableOp╩
Conv1D_5_256/BiasAddBiasAdd$Conv1D_5_256/conv1d/Squeeze:output:0+Conv1D_5_256/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:                  А2
Conv1D_5_256/BiasAddЦ
Conv1D_5_256/SigmoidSigmoidConv1D_5_256/BiasAdd:output:0*
T0*5
_output_shapes#
!:                  А2
Conv1D_5_256/Sigmoidд
Conv1D_5_256/mulMulConv1D_5_256/BiasAdd:output:0Conv1D_5_256/Sigmoid:y:0*
T0*5
_output_shapes#
!:                  А2
Conv1D_5_256/mulР
Conv1D_5_256/IdentityIdentityConv1D_5_256/mul:z:0*
T0*5
_output_shapes#
!:                  А2
Conv1D_5_256/IdentityЙ
Conv1D_5_256/IdentityN	IdentityNConv1D_5_256/mul:z:0Conv1D_5_256/BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1294350501*V
_output_shapesD
B:                  А:                  А2
Conv1D_5_256/IdentityNУ
"Conv1D_3_256/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2$
"Conv1D_3_256/conv1d/ExpandDims/dimх
Conv1D_3_256/conv1d/ExpandDims
ExpandDims$Batch_Norm_3_128/batchnorm/add_1:z:0+Conv1D_3_256/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#                  А2 
Conv1D_3_256/conv1d/ExpandDimsс
/Conv1D_3_256/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp8conv1d_3_256_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:АА*
dtype021
/Conv1D_3_256/conv1d/ExpandDims_1/ReadVariableOpО
$Conv1D_3_256/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Conv1D_3_256/conv1d/ExpandDims_1/dimэ
 Conv1D_3_256/conv1d/ExpandDims_1
ExpandDims7Conv1D_3_256/conv1d/ExpandDims_1/ReadVariableOp:value:0-Conv1D_3_256/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:АА2"
 Conv1D_3_256/conv1d/ExpandDims_1Ї
Conv1D_3_256/conv1dConv2D'Conv1D_3_256/conv1d/ExpandDims:output:0)Conv1D_3_256/conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#                  А*
paddingSAME*
strides
2
Conv1D_3_256/conv1d├
Conv1D_3_256/conv1d/SqueezeSqueezeConv1D_3_256/conv1d:output:0*
T0*5
_output_shapes#
!:                  А*
squeeze_dims

¤        2
Conv1D_3_256/conv1d/Squeeze┤
#Conv1D_3_256/BiasAdd/ReadVariableOpReadVariableOp,conv1d_3_256_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#Conv1D_3_256/BiasAdd/ReadVariableOp╩
Conv1D_3_256/BiasAddBiasAdd$Conv1D_3_256/conv1d/Squeeze:output:0+Conv1D_3_256/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:                  А2
Conv1D_3_256/BiasAddЦ
Conv1D_3_256/SigmoidSigmoidConv1D_3_256/BiasAdd:output:0*
T0*5
_output_shapes#
!:                  А2
Conv1D_3_256/Sigmoidд
Conv1D_3_256/mulMulConv1D_3_256/BiasAdd:output:0Conv1D_3_256/Sigmoid:y:0*
T0*5
_output_shapes#
!:                  А2
Conv1D_3_256/mulР
Conv1D_3_256/IdentityIdentityConv1D_3_256/mul:z:0*
T0*5
_output_shapes#
!:                  А2
Conv1D_3_256/IdentityЙ
Conv1D_3_256/IdentityN	IdentityNConv1D_3_256/mul:z:0Conv1D_3_256/BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1294350518*V
_output_shapesD
B:                  А:                  А2
Conv1D_3_256/IdentityN│
/Batch_Norm_7_256/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       21
/Batch_Norm_7_256/moments/mean/reduction_indicesр
Batch_Norm_7_256/moments/meanMeanConv1D_7_256/IdentityN:output:08Batch_Norm_7_256/moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
Batch_Norm_7_256/moments/mean┤
%Batch_Norm_7_256/moments/StopGradientStopGradient&Batch_Norm_7_256/moments/mean:output:0*
T0*#
_output_shapes
:А2'
%Batch_Norm_7_256/moments/StopGradient■
*Batch_Norm_7_256/moments/SquaredDifferenceSquaredDifferenceConv1D_7_256/IdentityN:output:0.Batch_Norm_7_256/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:                  А2,
*Batch_Norm_7_256/moments/SquaredDifference╗
3Batch_Norm_7_256/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       25
3Batch_Norm_7_256/moments/variance/reduction_indices√
!Batch_Norm_7_256/moments/varianceMean.Batch_Norm_7_256/moments/SquaredDifference:z:0<Batch_Norm_7_256/moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2#
!Batch_Norm_7_256/moments/variance╡
 Batch_Norm_7_256/moments/SqueezeSqueeze&Batch_Norm_7_256/moments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2"
 Batch_Norm_7_256/moments/Squeeze╜
"Batch_Norm_7_256/moments/Squeeze_1Squeeze*Batch_Norm_7_256/moments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2$
"Batch_Norm_7_256/moments/Squeeze_1Х
&Batch_Norm_7_256/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2(
&Batch_Norm_7_256/AssignMovingAvg/decay╪
/Batch_Norm_7_256/AssignMovingAvg/ReadVariableOpReadVariableOp8batch_norm_7_256_assignmovingavg_readvariableop_resource*
_output_shapes	
:А*
dtype021
/Batch_Norm_7_256/AssignMovingAvg/ReadVariableOp▌
$Batch_Norm_7_256/AssignMovingAvg/subSub7Batch_Norm_7_256/AssignMovingAvg/ReadVariableOp:value:0)Batch_Norm_7_256/moments/Squeeze:output:0*
T0*
_output_shapes	
:А2&
$Batch_Norm_7_256/AssignMovingAvg/sub╘
$Batch_Norm_7_256/AssignMovingAvg/mulMul(Batch_Norm_7_256/AssignMovingAvg/sub:z:0/Batch_Norm_7_256/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:А2&
$Batch_Norm_7_256/AssignMovingAvg/mulФ
 Batch_Norm_7_256/AssignMovingAvgAssignSubVariableOp8batch_norm_7_256_assignmovingavg_readvariableop_resource(Batch_Norm_7_256/AssignMovingAvg/mul:z:00^Batch_Norm_7_256/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02"
 Batch_Norm_7_256/AssignMovingAvgЩ
(Batch_Norm_7_256/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2*
(Batch_Norm_7_256/AssignMovingAvg_1/decay▐
1Batch_Norm_7_256/AssignMovingAvg_1/ReadVariableOpReadVariableOp:batch_norm_7_256_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:А*
dtype023
1Batch_Norm_7_256/AssignMovingAvg_1/ReadVariableOpх
&Batch_Norm_7_256/AssignMovingAvg_1/subSub9Batch_Norm_7_256/AssignMovingAvg_1/ReadVariableOp:value:0+Batch_Norm_7_256/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:А2(
&Batch_Norm_7_256/AssignMovingAvg_1/sub▄
&Batch_Norm_7_256/AssignMovingAvg_1/mulMul*Batch_Norm_7_256/AssignMovingAvg_1/sub:z:01Batch_Norm_7_256/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:А2(
&Batch_Norm_7_256/AssignMovingAvg_1/mulЮ
"Batch_Norm_7_256/AssignMovingAvg_1AssignSubVariableOp:batch_norm_7_256_assignmovingavg_1_readvariableop_resource*Batch_Norm_7_256/AssignMovingAvg_1/mul:z:02^Batch_Norm_7_256/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02$
"Batch_Norm_7_256/AssignMovingAvg_1Й
 Batch_Norm_7_256/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2"
 Batch_Norm_7_256/batchnorm/add/y╟
Batch_Norm_7_256/batchnorm/addAddV2+Batch_Norm_7_256/moments/Squeeze_1:output:0)Batch_Norm_7_256/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2 
Batch_Norm_7_256/batchnorm/addЧ
 Batch_Norm_7_256/batchnorm/RsqrtRsqrt"Batch_Norm_7_256/batchnorm/add:z:0*
T0*
_output_shapes	
:А2"
 Batch_Norm_7_256/batchnorm/Rsqrt╥
-Batch_Norm_7_256/batchnorm/mul/ReadVariableOpReadVariableOp6batch_norm_7_256_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-Batch_Norm_7_256/batchnorm/mul/ReadVariableOp╩
Batch_Norm_7_256/batchnorm/mulMul$Batch_Norm_7_256/batchnorm/Rsqrt:y:05Batch_Norm_7_256/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2 
Batch_Norm_7_256/batchnorm/mul╨
 Batch_Norm_7_256/batchnorm/mul_1MulConv1D_7_256/IdentityN:output:0"Batch_Norm_7_256/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2"
 Batch_Norm_7_256/batchnorm/mul_1└
 Batch_Norm_7_256/batchnorm/mul_2Mul)Batch_Norm_7_256/moments/Squeeze:output:0"Batch_Norm_7_256/batchnorm/mul:z:0*
T0*
_output_shapes	
:А2"
 Batch_Norm_7_256/batchnorm/mul_2╞
)Batch_Norm_7_256/batchnorm/ReadVariableOpReadVariableOp2batch_norm_7_256_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)Batch_Norm_7_256/batchnorm/ReadVariableOp╞
Batch_Norm_7_256/batchnorm/subSub1Batch_Norm_7_256/batchnorm/ReadVariableOp:value:0$Batch_Norm_7_256/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2 
Batch_Norm_7_256/batchnorm/sub╫
 Batch_Norm_7_256/batchnorm/add_1AddV2$Batch_Norm_7_256/batchnorm/mul_1:z:0"Batch_Norm_7_256/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2"
 Batch_Norm_7_256/batchnorm/add_1│
/Batch_Norm_5_256/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       21
/Batch_Norm_5_256/moments/mean/reduction_indicesр
Batch_Norm_5_256/moments/meanMeanConv1D_5_256/IdentityN:output:08Batch_Norm_5_256/moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
Batch_Norm_5_256/moments/mean┤
%Batch_Norm_5_256/moments/StopGradientStopGradient&Batch_Norm_5_256/moments/mean:output:0*
T0*#
_output_shapes
:А2'
%Batch_Norm_5_256/moments/StopGradient■
*Batch_Norm_5_256/moments/SquaredDifferenceSquaredDifferenceConv1D_5_256/IdentityN:output:0.Batch_Norm_5_256/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:                  А2,
*Batch_Norm_5_256/moments/SquaredDifference╗
3Batch_Norm_5_256/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       25
3Batch_Norm_5_256/moments/variance/reduction_indices√
!Batch_Norm_5_256/moments/varianceMean.Batch_Norm_5_256/moments/SquaredDifference:z:0<Batch_Norm_5_256/moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2#
!Batch_Norm_5_256/moments/variance╡
 Batch_Norm_5_256/moments/SqueezeSqueeze&Batch_Norm_5_256/moments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2"
 Batch_Norm_5_256/moments/Squeeze╜
"Batch_Norm_5_256/moments/Squeeze_1Squeeze*Batch_Norm_5_256/moments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2$
"Batch_Norm_5_256/moments/Squeeze_1Х
&Batch_Norm_5_256/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2(
&Batch_Norm_5_256/AssignMovingAvg/decay╪
/Batch_Norm_5_256/AssignMovingAvg/ReadVariableOpReadVariableOp8batch_norm_5_256_assignmovingavg_readvariableop_resource*
_output_shapes	
:А*
dtype021
/Batch_Norm_5_256/AssignMovingAvg/ReadVariableOp▌
$Batch_Norm_5_256/AssignMovingAvg/subSub7Batch_Norm_5_256/AssignMovingAvg/ReadVariableOp:value:0)Batch_Norm_5_256/moments/Squeeze:output:0*
T0*
_output_shapes	
:А2&
$Batch_Norm_5_256/AssignMovingAvg/sub╘
$Batch_Norm_5_256/AssignMovingAvg/mulMul(Batch_Norm_5_256/AssignMovingAvg/sub:z:0/Batch_Norm_5_256/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:А2&
$Batch_Norm_5_256/AssignMovingAvg/mulФ
 Batch_Norm_5_256/AssignMovingAvgAssignSubVariableOp8batch_norm_5_256_assignmovingavg_readvariableop_resource(Batch_Norm_5_256/AssignMovingAvg/mul:z:00^Batch_Norm_5_256/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02"
 Batch_Norm_5_256/AssignMovingAvgЩ
(Batch_Norm_5_256/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2*
(Batch_Norm_5_256/AssignMovingAvg_1/decay▐
1Batch_Norm_5_256/AssignMovingAvg_1/ReadVariableOpReadVariableOp:batch_norm_5_256_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:А*
dtype023
1Batch_Norm_5_256/AssignMovingAvg_1/ReadVariableOpх
&Batch_Norm_5_256/AssignMovingAvg_1/subSub9Batch_Norm_5_256/AssignMovingAvg_1/ReadVariableOp:value:0+Batch_Norm_5_256/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:А2(
&Batch_Norm_5_256/AssignMovingAvg_1/sub▄
&Batch_Norm_5_256/AssignMovingAvg_1/mulMul*Batch_Norm_5_256/AssignMovingAvg_1/sub:z:01Batch_Norm_5_256/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:А2(
&Batch_Norm_5_256/AssignMovingAvg_1/mulЮ
"Batch_Norm_5_256/AssignMovingAvg_1AssignSubVariableOp:batch_norm_5_256_assignmovingavg_1_readvariableop_resource*Batch_Norm_5_256/AssignMovingAvg_1/mul:z:02^Batch_Norm_5_256/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02$
"Batch_Norm_5_256/AssignMovingAvg_1Й
 Batch_Norm_5_256/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2"
 Batch_Norm_5_256/batchnorm/add/y╟
Batch_Norm_5_256/batchnorm/addAddV2+Batch_Norm_5_256/moments/Squeeze_1:output:0)Batch_Norm_5_256/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2 
Batch_Norm_5_256/batchnorm/addЧ
 Batch_Norm_5_256/batchnorm/RsqrtRsqrt"Batch_Norm_5_256/batchnorm/add:z:0*
T0*
_output_shapes	
:А2"
 Batch_Norm_5_256/batchnorm/Rsqrt╥
-Batch_Norm_5_256/batchnorm/mul/ReadVariableOpReadVariableOp6batch_norm_5_256_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-Batch_Norm_5_256/batchnorm/mul/ReadVariableOp╩
Batch_Norm_5_256/batchnorm/mulMul$Batch_Norm_5_256/batchnorm/Rsqrt:y:05Batch_Norm_5_256/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2 
Batch_Norm_5_256/batchnorm/mul╨
 Batch_Norm_5_256/batchnorm/mul_1MulConv1D_5_256/IdentityN:output:0"Batch_Norm_5_256/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2"
 Batch_Norm_5_256/batchnorm/mul_1└
 Batch_Norm_5_256/batchnorm/mul_2Mul)Batch_Norm_5_256/moments/Squeeze:output:0"Batch_Norm_5_256/batchnorm/mul:z:0*
T0*
_output_shapes	
:А2"
 Batch_Norm_5_256/batchnorm/mul_2╞
)Batch_Norm_5_256/batchnorm/ReadVariableOpReadVariableOp2batch_norm_5_256_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)Batch_Norm_5_256/batchnorm/ReadVariableOp╞
Batch_Norm_5_256/batchnorm/subSub1Batch_Norm_5_256/batchnorm/ReadVariableOp:value:0$Batch_Norm_5_256/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2 
Batch_Norm_5_256/batchnorm/sub╫
 Batch_Norm_5_256/batchnorm/add_1AddV2$Batch_Norm_5_256/batchnorm/mul_1:z:0"Batch_Norm_5_256/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2"
 Batch_Norm_5_256/batchnorm/add_1│
/Batch_Norm_3_256/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       21
/Batch_Norm_3_256/moments/mean/reduction_indicesр
Batch_Norm_3_256/moments/meanMeanConv1D_3_256/IdentityN:output:08Batch_Norm_3_256/moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
Batch_Norm_3_256/moments/mean┤
%Batch_Norm_3_256/moments/StopGradientStopGradient&Batch_Norm_3_256/moments/mean:output:0*
T0*#
_output_shapes
:А2'
%Batch_Norm_3_256/moments/StopGradient■
*Batch_Norm_3_256/moments/SquaredDifferenceSquaredDifferenceConv1D_3_256/IdentityN:output:0.Batch_Norm_3_256/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:                  А2,
*Batch_Norm_3_256/moments/SquaredDifference╗
3Batch_Norm_3_256/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       25
3Batch_Norm_3_256/moments/variance/reduction_indices√
!Batch_Norm_3_256/moments/varianceMean.Batch_Norm_3_256/moments/SquaredDifference:z:0<Batch_Norm_3_256/moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2#
!Batch_Norm_3_256/moments/variance╡
 Batch_Norm_3_256/moments/SqueezeSqueeze&Batch_Norm_3_256/moments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2"
 Batch_Norm_3_256/moments/Squeeze╜
"Batch_Norm_3_256/moments/Squeeze_1Squeeze*Batch_Norm_3_256/moments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2$
"Batch_Norm_3_256/moments/Squeeze_1Х
&Batch_Norm_3_256/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2(
&Batch_Norm_3_256/AssignMovingAvg/decay╪
/Batch_Norm_3_256/AssignMovingAvg/ReadVariableOpReadVariableOp8batch_norm_3_256_assignmovingavg_readvariableop_resource*
_output_shapes	
:А*
dtype021
/Batch_Norm_3_256/AssignMovingAvg/ReadVariableOp▌
$Batch_Norm_3_256/AssignMovingAvg/subSub7Batch_Norm_3_256/AssignMovingAvg/ReadVariableOp:value:0)Batch_Norm_3_256/moments/Squeeze:output:0*
T0*
_output_shapes	
:А2&
$Batch_Norm_3_256/AssignMovingAvg/sub╘
$Batch_Norm_3_256/AssignMovingAvg/mulMul(Batch_Norm_3_256/AssignMovingAvg/sub:z:0/Batch_Norm_3_256/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:А2&
$Batch_Norm_3_256/AssignMovingAvg/mulФ
 Batch_Norm_3_256/AssignMovingAvgAssignSubVariableOp8batch_norm_3_256_assignmovingavg_readvariableop_resource(Batch_Norm_3_256/AssignMovingAvg/mul:z:00^Batch_Norm_3_256/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02"
 Batch_Norm_3_256/AssignMovingAvgЩ
(Batch_Norm_3_256/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2*
(Batch_Norm_3_256/AssignMovingAvg_1/decay▐
1Batch_Norm_3_256/AssignMovingAvg_1/ReadVariableOpReadVariableOp:batch_norm_3_256_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:А*
dtype023
1Batch_Norm_3_256/AssignMovingAvg_1/ReadVariableOpх
&Batch_Norm_3_256/AssignMovingAvg_1/subSub9Batch_Norm_3_256/AssignMovingAvg_1/ReadVariableOp:value:0+Batch_Norm_3_256/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:А2(
&Batch_Norm_3_256/AssignMovingAvg_1/sub▄
&Batch_Norm_3_256/AssignMovingAvg_1/mulMul*Batch_Norm_3_256/AssignMovingAvg_1/sub:z:01Batch_Norm_3_256/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:А2(
&Batch_Norm_3_256/AssignMovingAvg_1/mulЮ
"Batch_Norm_3_256/AssignMovingAvg_1AssignSubVariableOp:batch_norm_3_256_assignmovingavg_1_readvariableop_resource*Batch_Norm_3_256/AssignMovingAvg_1/mul:z:02^Batch_Norm_3_256/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02$
"Batch_Norm_3_256/AssignMovingAvg_1Й
 Batch_Norm_3_256/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2"
 Batch_Norm_3_256/batchnorm/add/y╟
Batch_Norm_3_256/batchnorm/addAddV2+Batch_Norm_3_256/moments/Squeeze_1:output:0)Batch_Norm_3_256/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2 
Batch_Norm_3_256/batchnorm/addЧ
 Batch_Norm_3_256/batchnorm/RsqrtRsqrt"Batch_Norm_3_256/batchnorm/add:z:0*
T0*
_output_shapes	
:А2"
 Batch_Norm_3_256/batchnorm/Rsqrt╥
-Batch_Norm_3_256/batchnorm/mul/ReadVariableOpReadVariableOp6batch_norm_3_256_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-Batch_Norm_3_256/batchnorm/mul/ReadVariableOp╩
Batch_Norm_3_256/batchnorm/mulMul$Batch_Norm_3_256/batchnorm/Rsqrt:y:05Batch_Norm_3_256/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2 
Batch_Norm_3_256/batchnorm/mul╨
 Batch_Norm_3_256/batchnorm/mul_1MulConv1D_3_256/IdentityN:output:0"Batch_Norm_3_256/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2"
 Batch_Norm_3_256/batchnorm/mul_1└
 Batch_Norm_3_256/batchnorm/mul_2Mul)Batch_Norm_3_256/moments/Squeeze:output:0"Batch_Norm_3_256/batchnorm/mul:z:0*
T0*
_output_shapes	
:А2"
 Batch_Norm_3_256/batchnorm/mul_2╞
)Batch_Norm_3_256/batchnorm/ReadVariableOpReadVariableOp2batch_norm_3_256_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)Batch_Norm_3_256/batchnorm/ReadVariableOp╞
Batch_Norm_3_256/batchnorm/subSub1Batch_Norm_3_256/batchnorm/ReadVariableOp:value:0$Batch_Norm_3_256/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2 
Batch_Norm_3_256/batchnorm/sub╫
 Batch_Norm_3_256/batchnorm/add_1AddV2$Batch_Norm_3_256/batchnorm/mul_1:z:0"Batch_Norm_3_256/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2"
 Batch_Norm_3_256/batchnorm/add_1Ж
 Max_Pool_3/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2"
 Max_Pool_3/Max/reduction_indicesл
Max_Pool_3/MaxMax$Batch_Norm_3_256/batchnorm/add_1:z:0)Max_Pool_3/Max/reduction_indices:output:0*
T0*(
_output_shapes
:         А2
Max_Pool_3/MaxЖ
 Max_Pool_5/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2"
 Max_Pool_5/Max/reduction_indicesл
Max_Pool_5/MaxMax$Batch_Norm_5_256/batchnorm/add_1:z:0)Max_Pool_5/Max/reduction_indices:output:0*
T0*(
_output_shapes
:         А2
Max_Pool_5/MaxЖ
 Max_Pool_7/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2"
 Max_Pool_7/Max/reduction_indicesл
Max_Pool_7/MaxMax$Batch_Norm_7_256/batchnorm/add_1:z:0)Max_Pool_7/Max/reduction_indices:output:0*
T0*(
_output_shapes
:         А2
Max_Pool_7/Max~
Pool_Concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
Pool_Concatenate/concat/axisь
Pool_Concatenate/concatConcatV2Max_Pool_3/Max:output:0Max_Pool_5/Max:output:0Max_Pool_7/Max:output:0%Pool_Concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:         А2
Pool_Concatenate/concatЛ
!Concatenate_Dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2#
!Concatenate_Dropout/dropout/Const╩
Concatenate_Dropout/dropout/MulMul Pool_Concatenate/concat:output:0*Concatenate_Dropout/dropout/Const:output:0*
T0*(
_output_shapes
:         А2!
Concatenate_Dropout/dropout/MulЦ
!Concatenate_Dropout/dropout/ShapeShape Pool_Concatenate/concat:output:0*
T0*
_output_shapes
:2#
!Concatenate_Dropout/dropout/Shapeё
8Concatenate_Dropout/dropout/random_uniform/RandomUniformRandomUniform*Concatenate_Dropout/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype02:
8Concatenate_Dropout/dropout/random_uniform/RandomUniformЭ
*Concatenate_Dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2,
*Concatenate_Dropout/dropout/GreaterEqual/yП
(Concatenate_Dropout/dropout/GreaterEqualGreaterEqualAConcatenate_Dropout/dropout/random_uniform/RandomUniform:output:03Concatenate_Dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2*
(Concatenate_Dropout/dropout/GreaterEqual╝
 Concatenate_Dropout/dropout/CastCast,Concatenate_Dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2"
 Concatenate_Dropout/dropout/Cast╦
!Concatenate_Dropout/dropout/Mul_1Mul#Concatenate_Dropout/dropout/Mul:z:0$Concatenate_Dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:         А2#
!Concatenate_Dropout/dropout/Mul_1п
 Prediction/MatMul/ReadVariableOpReadVariableOp)prediction_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02"
 Prediction/MatMul/ReadVariableOp│
Prediction/MatMulMatMul%Concatenate_Dropout/dropout/Mul_1:z:0(Prediction/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
Prediction/MatMulн
!Prediction/BiasAdd/ReadVariableOpReadVariableOp*prediction_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!Prediction/BiasAdd/ReadVariableOpн
Prediction/BiasAddBiasAddPrediction/MatMul:product:0)Prediction/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
Prediction/BiasAddВ
Prediction/SigmoidSigmoidPrediction/BiasAdd:output:0*
T0*'
_output_shapes
:         2
Prediction/Sigmoid¤
IdentityIdentityPrediction/Sigmoid:y:0!^Batch_Norm_3_128/AssignMovingAvg0^Batch_Norm_3_128/AssignMovingAvg/ReadVariableOp#^Batch_Norm_3_128/AssignMovingAvg_12^Batch_Norm_3_128/AssignMovingAvg_1/ReadVariableOp*^Batch_Norm_3_128/batchnorm/ReadVariableOp.^Batch_Norm_3_128/batchnorm/mul/ReadVariableOp!^Batch_Norm_3_256/AssignMovingAvg0^Batch_Norm_3_256/AssignMovingAvg/ReadVariableOp#^Batch_Norm_3_256/AssignMovingAvg_12^Batch_Norm_3_256/AssignMovingAvg_1/ReadVariableOp*^Batch_Norm_3_256/batchnorm/ReadVariableOp.^Batch_Norm_3_256/batchnorm/mul/ReadVariableOp!^Batch_Norm_5_128/AssignMovingAvg0^Batch_Norm_5_128/AssignMovingAvg/ReadVariableOp#^Batch_Norm_5_128/AssignMovingAvg_12^Batch_Norm_5_128/AssignMovingAvg_1/ReadVariableOp*^Batch_Norm_5_128/batchnorm/ReadVariableOp.^Batch_Norm_5_128/batchnorm/mul/ReadVariableOp!^Batch_Norm_5_256/AssignMovingAvg0^Batch_Norm_5_256/AssignMovingAvg/ReadVariableOp#^Batch_Norm_5_256/AssignMovingAvg_12^Batch_Norm_5_256/AssignMovingAvg_1/ReadVariableOp*^Batch_Norm_5_256/batchnorm/ReadVariableOp.^Batch_Norm_5_256/batchnorm/mul/ReadVariableOp!^Batch_Norm_7_128/AssignMovingAvg0^Batch_Norm_7_128/AssignMovingAvg/ReadVariableOp#^Batch_Norm_7_128/AssignMovingAvg_12^Batch_Norm_7_128/AssignMovingAvg_1/ReadVariableOp*^Batch_Norm_7_128/batchnorm/ReadVariableOp.^Batch_Norm_7_128/batchnorm/mul/ReadVariableOp!^Batch_Norm_7_256/AssignMovingAvg0^Batch_Norm_7_256/AssignMovingAvg/ReadVariableOp#^Batch_Norm_7_256/AssignMovingAvg_12^Batch_Norm_7_256/AssignMovingAvg_1/ReadVariableOp*^Batch_Norm_7_256/batchnorm/ReadVariableOp.^Batch_Norm_7_256/batchnorm/mul/ReadVariableOp$^Conv1D_3_128/BiasAdd/ReadVariableOp0^Conv1D_3_128/conv1d/ExpandDims_1/ReadVariableOp$^Conv1D_3_256/BiasAdd/ReadVariableOp0^Conv1D_3_256/conv1d/ExpandDims_1/ReadVariableOp$^Conv1D_5_128/BiasAdd/ReadVariableOp0^Conv1D_5_128/conv1d/ExpandDims_1/ReadVariableOp$^Conv1D_5_256/BiasAdd/ReadVariableOp0^Conv1D_5_256/conv1d/ExpandDims_1/ReadVariableOp$^Conv1D_7_128/BiasAdd/ReadVariableOp0^Conv1D_7_128/conv1d/ExpandDims_1/ReadVariableOp$^Conv1D_7_256/BiasAdd/ReadVariableOp0^Conv1D_7_256/conv1d/ExpandDims_1/ReadVariableOp"^Prediction/BiasAdd/ReadVariableOp!^Prediction/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*А
_input_shapeso
m:                  А: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
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
!:                  А
 
_user_specified_nameinputs
№
а
L__inference_Conv1D_7_128_layer_call_and_return_conditional_losses_1294348881

inputsC
+conv1d_expanddims_1_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А

identity_1ИвBiasAdd/ReadVariableOpв"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimа
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#                  А2
conv1d/ExpandDims║
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:АА*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim╣
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:АА2
conv1d/ExpandDims_1└
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#                  А*
paddingSAME*
strides
2
conv1dЬ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*5
_output_shapes#
!:                  А*
squeeze_dims

¤        2
conv1d/SqueezeН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЦ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:                  А2	
BiasAddo
SigmoidSigmoidBiasAdd:output:0*
T0*5
_output_shapes#
!:                  А2	
Sigmoidp
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*5
_output_shapes#
!:                  А2
muli
IdentityIdentitymul:z:0*
T0*5
_output_shapes#
!:                  А2

Identity╒
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1294348874*V
_output_shapesD
B:                  А:                  А2
	IdentityN╢

Identity_1IdentityIdentityN:output:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*5
_output_shapes#
!:                  А2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:                  А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
╕

№
J__inference_Prediction_layer_call_and_return_conditional_losses_1294349107

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         2	
SigmoidР
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
ё
│
P__inference_Batch_Norm_5_128_layer_call_and_return_conditional_losses_1294348021

inputs0
!batchnorm_readvariableop_resource:	А4
%batchnorm_mul_readvariableop_resource:	А2
#batchnorm_readvariableop_1_resource:	А2
#batchnorm_readvariableop_2_resource:	А
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/add_1щ
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  А: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
э
е
1__inference_Conv1D_5_128_layer_call_fn_1294350704

inputs
unknown:АА
	unknown_0:	А
identityИвStatefulPartitionedCallН
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_Conv1D_5_128_layer_call_and_return_conditional_losses_12943489082
StatefulPartitionedCallЬ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:                  А: : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
Э
o
6__inference_Embedding_Dropout_layer_call_fn_1294350648

inputs
identityИвStatefulPartitionedCall°
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_Embedding_Dropout_layer_call_and_return_conditional_losses_12943493142
StatefulPartitionedCallЬ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:                  А22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
№
а
L__inference_Conv1D_5_256_layer_call_and_return_conditional_losses_1294349016

inputsC
+conv1d_expanddims_1_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А

identity_1ИвBiasAdd/ReadVariableOpв"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimа
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#                  А2
conv1d/ExpandDims║
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:АА*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim╣
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:АА2
conv1d/ExpandDims_1└
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#                  А*
paddingSAME*
strides
2
conv1dЬ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*5
_output_shapes#
!:                  А*
squeeze_dims

¤        2
conv1d/SqueezeН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЦ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:                  А2	
BiasAddo
SigmoidSigmoidBiasAdd:output:0*
T0*5
_output_shapes#
!:                  А2	
Sigmoidp
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*5
_output_shapes#
!:                  А2
muli
IdentityIdentitymul:z:0*
T0*5
_output_shapes#
!:                  А2

Identity╒
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1294349009*V
_output_shapesD
B:                  А:                  А2
	IdentityN╢

Identity_1IdentityIdentityN:output:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*5
_output_shapes#
!:                  А2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:                  А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
яn
█
L__inference_Tree_Encoder_layer_call_and_return_conditional_losses_1294349499

inputs/
conv1d_7_128_1294349404:АА&
conv1d_7_128_1294349406:	А/
conv1d_5_128_1294349409:АА&
conv1d_5_128_1294349411:	А/
conv1d_3_128_1294349414:АА&
conv1d_3_128_1294349416:	А*
batch_norm_7_128_1294349419:	А*
batch_norm_7_128_1294349421:	А*
batch_norm_7_128_1294349423:	А*
batch_norm_7_128_1294349425:	А*
batch_norm_5_128_1294349428:	А*
batch_norm_5_128_1294349430:	А*
batch_norm_5_128_1294349432:	А*
batch_norm_5_128_1294349434:	А*
batch_norm_3_128_1294349437:	А*
batch_norm_3_128_1294349439:	А*
batch_norm_3_128_1294349441:	А*
batch_norm_3_128_1294349443:	А/
conv1d_7_256_1294349446:АА&
conv1d_7_256_1294349448:	А/
conv1d_5_256_1294349451:АА&
conv1d_5_256_1294349453:	А/
conv1d_3_256_1294349456:АА&
conv1d_3_256_1294349458:	А*
batch_norm_7_256_1294349461:	А*
batch_norm_7_256_1294349463:	А*
batch_norm_7_256_1294349465:	А*
batch_norm_7_256_1294349467:	А*
batch_norm_5_256_1294349470:	А*
batch_norm_5_256_1294349472:	А*
batch_norm_5_256_1294349474:	А*
batch_norm_5_256_1294349476:	А*
batch_norm_3_256_1294349479:	А*
batch_norm_3_256_1294349481:	А*
batch_norm_3_256_1294349483:	А*
batch_norm_3_256_1294349485:	А(
prediction_1294349493:	А#
prediction_1294349495:
identityИв(Batch_Norm_3_128/StatefulPartitionedCallв(Batch_Norm_3_256/StatefulPartitionedCallв(Batch_Norm_5_128/StatefulPartitionedCallв(Batch_Norm_5_256/StatefulPartitionedCallв(Batch_Norm_7_128/StatefulPartitionedCallв(Batch_Norm_7_256/StatefulPartitionedCallв+Concatenate_Dropout/StatefulPartitionedCallв$Conv1D_3_128/StatefulPartitionedCallв$Conv1D_3_256/StatefulPartitionedCallв$Conv1D_5_128/StatefulPartitionedCallв$Conv1D_5_256/StatefulPartitionedCallв$Conv1D_7_128/StatefulPartitionedCallв$Conv1D_7_256/StatefulPartitionedCallв)Embedding_Dropout/StatefulPartitionedCallв"Prediction/StatefulPartitionedCallЬ
)Embedding_Dropout/StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_Embedding_Dropout_layer_call_and_return_conditional_losses_12943493142+
)Embedding_Dropout/StatefulPartitionedCallё
$Conv1D_7_128/StatefulPartitionedCallStatefulPartitionedCall2Embedding_Dropout/StatefulPartitionedCall:output:0conv1d_7_128_1294349404conv1d_7_128_1294349406*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_Conv1D_7_128_layer_call_and_return_conditional_losses_12943488812&
$Conv1D_7_128/StatefulPartitionedCallё
$Conv1D_5_128/StatefulPartitionedCallStatefulPartitionedCall2Embedding_Dropout/StatefulPartitionedCall:output:0conv1d_5_128_1294349409conv1d_5_128_1294349411*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_Conv1D_5_128_layer_call_and_return_conditional_losses_12943489082&
$Conv1D_5_128/StatefulPartitionedCallё
$Conv1D_3_128/StatefulPartitionedCallStatefulPartitionedCall2Embedding_Dropout/StatefulPartitionedCall:output:0conv1d_3_128_1294349414conv1d_3_128_1294349416*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_Conv1D_3_128_layer_call_and_return_conditional_losses_12943489352&
$Conv1D_3_128/StatefulPartitionedCall╝
(Batch_Norm_7_128/StatefulPartitionedCallStatefulPartitionedCall-Conv1D_7_128/StatefulPartitionedCall:output:0batch_norm_7_128_1294349419batch_norm_7_128_1294349421batch_norm_7_128_1294349423batch_norm_7_128_1294349425*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_Batch_Norm_7_128_layer_call_and_return_conditional_losses_12943482432*
(Batch_Norm_7_128/StatefulPartitionedCall╝
(Batch_Norm_5_128/StatefulPartitionedCallStatefulPartitionedCall-Conv1D_5_128/StatefulPartitionedCall:output:0batch_norm_5_128_1294349428batch_norm_5_128_1294349430batch_norm_5_128_1294349432batch_norm_5_128_1294349434*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_Batch_Norm_5_128_layer_call_and_return_conditional_losses_12943480812*
(Batch_Norm_5_128/StatefulPartitionedCall╝
(Batch_Norm_3_128/StatefulPartitionedCallStatefulPartitionedCall-Conv1D_3_128/StatefulPartitionedCall:output:0batch_norm_3_128_1294349437batch_norm_3_128_1294349439batch_norm_3_128_1294349441batch_norm_3_128_1294349443*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_Batch_Norm_3_128_layer_call_and_return_conditional_losses_12943479192*
(Batch_Norm_3_128/StatefulPartitionedCallЁ
$Conv1D_7_256/StatefulPartitionedCallStatefulPartitionedCall1Batch_Norm_7_128/StatefulPartitionedCall:output:0conv1d_7_256_1294349446conv1d_7_256_1294349448*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_Conv1D_7_256_layer_call_and_return_conditional_losses_12943489892&
$Conv1D_7_256/StatefulPartitionedCallЁ
$Conv1D_5_256/StatefulPartitionedCallStatefulPartitionedCall1Batch_Norm_5_128/StatefulPartitionedCall:output:0conv1d_5_256_1294349451conv1d_5_256_1294349453*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_Conv1D_5_256_layer_call_and_return_conditional_losses_12943490162&
$Conv1D_5_256/StatefulPartitionedCallЁ
$Conv1D_3_256/StatefulPartitionedCallStatefulPartitionedCall1Batch_Norm_3_128/StatefulPartitionedCall:output:0conv1d_3_256_1294349456conv1d_3_256_1294349458*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_Conv1D_3_256_layer_call_and_return_conditional_losses_12943490432&
$Conv1D_3_256/StatefulPartitionedCall╝
(Batch_Norm_7_256/StatefulPartitionedCallStatefulPartitionedCall-Conv1D_7_256/StatefulPartitionedCall:output:0batch_norm_7_256_1294349461batch_norm_7_256_1294349463batch_norm_7_256_1294349465batch_norm_7_256_1294349467*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_Batch_Norm_7_256_layer_call_and_return_conditional_losses_12943487292*
(Batch_Norm_7_256/StatefulPartitionedCall╝
(Batch_Norm_5_256/StatefulPartitionedCallStatefulPartitionedCall-Conv1D_5_256/StatefulPartitionedCall:output:0batch_norm_5_256_1294349470batch_norm_5_256_1294349472batch_norm_5_256_1294349474batch_norm_5_256_1294349476*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_Batch_Norm_5_256_layer_call_and_return_conditional_losses_12943485672*
(Batch_Norm_5_256/StatefulPartitionedCall╝
(Batch_Norm_3_256/StatefulPartitionedCallStatefulPartitionedCall-Conv1D_3_256/StatefulPartitionedCall:output:0batch_norm_3_256_1294349479batch_norm_3_256_1294349481batch_norm_3_256_1294349483batch_norm_3_256_1294349485*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_Batch_Norm_3_256_layer_call_and_return_conditional_losses_12943484052*
(Batch_Norm_3_256/StatefulPartitionedCallН
Max_Pool_3/PartitionedCallPartitionedCall1Batch_Norm_3_256/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_Max_Pool_3_layer_call_and_return_conditional_losses_12943488142
Max_Pool_3/PartitionedCallН
Max_Pool_5/PartitionedCallPartitionedCall1Batch_Norm_5_256/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_Max_Pool_5_layer_call_and_return_conditional_losses_12943488272
Max_Pool_5/PartitionedCallН
Max_Pool_7/PartitionedCallPartitionedCall1Batch_Norm_7_256/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_Max_Pool_7_layer_call_and_return_conditional_losses_12943488402
Max_Pool_7/PartitionedCall▌
 Pool_Concatenate/PartitionedCallPartitionedCall#Max_Pool_3/PartitionedCall:output:0#Max_Pool_5/PartitionedCall:output:0#Max_Pool_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_Pool_Concatenate_layer_call_and_return_conditional_losses_12943490872"
 Pool_Concatenate/PartitionedCallф
+Concatenate_Dropout/StatefulPartitionedCallStatefulPartitionedCall)Pool_Concatenate/PartitionedCall:output:0*^Embedding_Dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_Concatenate_Dropout_layer_call_and_return_conditional_losses_12943492232-
+Concatenate_Dropout/StatefulPartitionedCall█
"Prediction/StatefulPartitionedCallStatefulPartitionedCall4Concatenate_Dropout/StatefulPartitionedCall:output:0prediction_1294349493prediction_1294349495*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_Prediction_layer_call_and_return_conditional_losses_12943491072$
"Prediction/StatefulPartitionedCallъ
IdentityIdentity+Prediction/StatefulPartitionedCall:output:0)^Batch_Norm_3_128/StatefulPartitionedCall)^Batch_Norm_3_256/StatefulPartitionedCall)^Batch_Norm_5_128/StatefulPartitionedCall)^Batch_Norm_5_256/StatefulPartitionedCall)^Batch_Norm_7_128/StatefulPartitionedCall)^Batch_Norm_7_256/StatefulPartitionedCall,^Concatenate_Dropout/StatefulPartitionedCall%^Conv1D_3_128/StatefulPartitionedCall%^Conv1D_3_256/StatefulPartitionedCall%^Conv1D_5_128/StatefulPartitionedCall%^Conv1D_5_256/StatefulPartitionedCall%^Conv1D_7_128/StatefulPartitionedCall%^Conv1D_7_256/StatefulPartitionedCall*^Embedding_Dropout/StatefulPartitionedCall#^Prediction/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*А
_input_shapeso
m:                  А: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2T
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
!:                  А
 
_user_specified_nameinputs
к
p
Q__inference_Embedding_Dropout_layer_call_and_return_conditional_losses_1294349314

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?2
dropout/ConstБ
dropout/MulMulinputsdropout/Const:output:0*
T0*5
_output_shapes#
!:                  А2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┬
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*5
_output_shapes#
!:                  А*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
dropout/GreaterEqual/y╠
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*5
_output_shapes#
!:                  А2
dropout/GreaterEqualН
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*5
_output_shapes#
!:                  А2
dropout/CastИ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*5
_output_shapes#
!:                  А2
dropout/Mul_1s
IdentityIdentitydropout/Mul_1:z:0*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:                  А:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
╝
╘
5__inference_Batch_Norm_3_256_layer_call_fn_1294351111

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_Batch_Norm_3_256_layer_call_and_return_conditional_losses_12943484052
StatefulPartitionedCallЬ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
№
Ш	
1__inference_Tree_Encoder_layer_call_fn_1294350021

inputs
unknown:АА
	unknown_0:	А!
	unknown_1:АА
	unknown_2:	А!
	unknown_3:АА
	unknown_4:	А
	unknown_5:	А
	unknown_6:	А
	unknown_7:	А
	unknown_8:	А
	unknown_9:	А

unknown_10:	А

unknown_11:	А

unknown_12:	А

unknown_13:	А

unknown_14:	А

unknown_15:	А

unknown_16:	А"

unknown_17:АА

unknown_18:	А"

unknown_19:АА

unknown_20:	А"

unknown_21:АА

unknown_22:	А

unknown_23:	А

unknown_24:	А

unknown_25:	А

unknown_26:	А

unknown_27:	А

unknown_28:	А

unknown_29:	А

unknown_30:	А

unknown_31:	А

unknown_32:	А

unknown_33:	А

unknown_34:	А

unknown_35:	А

unknown_36:
identityИвStatefulPartitionedCallю
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
:         *H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_Tree_Encoder_layer_call_and_return_conditional_losses_12943491142
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*А
_input_shapeso
m:                  А: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
Яд
Т
&__inference__traced_restore_1294351648
file_prefix<
$assignvariableop_conv1d_3_128_kernel:АА3
$assignvariableop_1_conv1d_3_128_bias:	А>
&assignvariableop_2_conv1d_5_128_kernel:АА3
$assignvariableop_3_conv1d_5_128_bias:	А>
&assignvariableop_4_conv1d_7_128_kernel:АА3
$assignvariableop_5_conv1d_7_128_bias:	А8
)assignvariableop_6_batch_norm_3_128_gamma:	А7
(assignvariableop_7_batch_norm_3_128_beta:	А>
/assignvariableop_8_batch_norm_3_128_moving_mean:	АB
3assignvariableop_9_batch_norm_3_128_moving_variance:	А9
*assignvariableop_10_batch_norm_5_128_gamma:	А8
)assignvariableop_11_batch_norm_5_128_beta:	А?
0assignvariableop_12_batch_norm_5_128_moving_mean:	АC
4assignvariableop_13_batch_norm_5_128_moving_variance:	А9
*assignvariableop_14_batch_norm_7_128_gamma:	А8
)assignvariableop_15_batch_norm_7_128_beta:	А?
0assignvariableop_16_batch_norm_7_128_moving_mean:	АC
4assignvariableop_17_batch_norm_7_128_moving_variance:	А?
'assignvariableop_18_conv1d_3_256_kernel:АА4
%assignvariableop_19_conv1d_3_256_bias:	А?
'assignvariableop_20_conv1d_5_256_kernel:АА4
%assignvariableop_21_conv1d_5_256_bias:	А?
'assignvariableop_22_conv1d_7_256_kernel:АА4
%assignvariableop_23_conv1d_7_256_bias:	А9
*assignvariableop_24_batch_norm_3_256_gamma:	А8
)assignvariableop_25_batch_norm_3_256_beta:	А?
0assignvariableop_26_batch_norm_3_256_moving_mean:	АC
4assignvariableop_27_batch_norm_3_256_moving_variance:	А9
*assignvariableop_28_batch_norm_5_256_gamma:	А8
)assignvariableop_29_batch_norm_5_256_beta:	А?
0assignvariableop_30_batch_norm_5_256_moving_mean:	АC
4assignvariableop_31_batch_norm_5_256_moving_variance:	А9
*assignvariableop_32_batch_norm_7_256_gamma:	А8
)assignvariableop_33_batch_norm_7_256_beta:	А?
0assignvariableop_34_batch_norm_7_256_moving_mean:	АC
4assignvariableop_35_batch_norm_7_256_moving_variance:	А8
%assignvariableop_36_prediction_kernel:	А1
#assignvariableop_37_prediction_bias:
identity_39ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_37вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9╡
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:'*
dtype0*┴
value╖B┤'B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names▄
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:'*
dtype0*a
valueXBV'B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesё
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*▓
_output_shapesЯ
Ь:::::::::::::::::::::::::::::::::::::::*5
dtypes+
)2'2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identityг
AssignVariableOpAssignVariableOp$assignvariableop_conv1d_3_128_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1й
AssignVariableOp_1AssignVariableOp$assignvariableop_1_conv1d_3_128_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2л
AssignVariableOp_2AssignVariableOp&assignvariableop_2_conv1d_5_128_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3й
AssignVariableOp_3AssignVariableOp$assignvariableop_3_conv1d_5_128_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4л
AssignVariableOp_4AssignVariableOp&assignvariableop_4_conv1d_7_128_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5й
AssignVariableOp_5AssignVariableOp$assignvariableop_5_conv1d_7_128_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6о
AssignVariableOp_6AssignVariableOp)assignvariableop_6_batch_norm_3_128_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7н
AssignVariableOp_7AssignVariableOp(assignvariableop_7_batch_norm_3_128_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8┤
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_norm_3_128_moving_meanIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9╕
AssignVariableOp_9AssignVariableOp3assignvariableop_9_batch_norm_3_128_moving_varianceIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10▓
AssignVariableOp_10AssignVariableOp*assignvariableop_10_batch_norm_5_128_gammaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11▒
AssignVariableOp_11AssignVariableOp)assignvariableop_11_batch_norm_5_128_betaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12╕
AssignVariableOp_12AssignVariableOp0assignvariableop_12_batch_norm_5_128_moving_meanIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13╝
AssignVariableOp_13AssignVariableOp4assignvariableop_13_batch_norm_5_128_moving_varianceIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14▓
AssignVariableOp_14AssignVariableOp*assignvariableop_14_batch_norm_7_128_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15▒
AssignVariableOp_15AssignVariableOp)assignvariableop_15_batch_norm_7_128_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16╕
AssignVariableOp_16AssignVariableOp0assignvariableop_16_batch_norm_7_128_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17╝
AssignVariableOp_17AssignVariableOp4assignvariableop_17_batch_norm_7_128_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18п
AssignVariableOp_18AssignVariableOp'assignvariableop_18_conv1d_3_256_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19н
AssignVariableOp_19AssignVariableOp%assignvariableop_19_conv1d_3_256_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20п
AssignVariableOp_20AssignVariableOp'assignvariableop_20_conv1d_5_256_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21н
AssignVariableOp_21AssignVariableOp%assignvariableop_21_conv1d_5_256_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22п
AssignVariableOp_22AssignVariableOp'assignvariableop_22_conv1d_7_256_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23н
AssignVariableOp_23AssignVariableOp%assignvariableop_23_conv1d_7_256_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24▓
AssignVariableOp_24AssignVariableOp*assignvariableop_24_batch_norm_3_256_gammaIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25▒
AssignVariableOp_25AssignVariableOp)assignvariableop_25_batch_norm_3_256_betaIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26╕
AssignVariableOp_26AssignVariableOp0assignvariableop_26_batch_norm_3_256_moving_meanIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27╝
AssignVariableOp_27AssignVariableOp4assignvariableop_27_batch_norm_3_256_moving_varianceIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28▓
AssignVariableOp_28AssignVariableOp*assignvariableop_28_batch_norm_5_256_gammaIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29▒
AssignVariableOp_29AssignVariableOp)assignvariableop_29_batch_norm_5_256_betaIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30╕
AssignVariableOp_30AssignVariableOp0assignvariableop_30_batch_norm_5_256_moving_meanIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31╝
AssignVariableOp_31AssignVariableOp4assignvariableop_31_batch_norm_5_256_moving_varianceIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32▓
AssignVariableOp_32AssignVariableOp*assignvariableop_32_batch_norm_7_256_gammaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33▒
AssignVariableOp_33AssignVariableOp)assignvariableop_33_batch_norm_7_256_betaIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34╕
AssignVariableOp_34AssignVariableOp0assignvariableop_34_batch_norm_7_256_moving_meanIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35╝
AssignVariableOp_35AssignVariableOp4assignvariableop_35_batch_norm_7_256_moving_varianceIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36н
AssignVariableOp_36AssignVariableOp%assignvariableop_36_prediction_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37л
AssignVariableOp_37AssignVariableOp#assignvariableop_37_prediction_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_379
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpв
Identity_38Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_38Х
Identity_39IdentityIdentity_38:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_39"#
identity_39Identity_39:output:0*a
_input_shapesP
N: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_37AssignVariableOp_372(
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
╕У
С*
%__inference__wrapped_model_1294347835

inputs]
Etree_encoder_conv1d_7_128_conv1d_expanddims_1_readvariableop_resource:ААH
9tree_encoder_conv1d_7_128_biasadd_readvariableop_resource:	А]
Etree_encoder_conv1d_5_128_conv1d_expanddims_1_readvariableop_resource:ААH
9tree_encoder_conv1d_5_128_biasadd_readvariableop_resource:	А]
Etree_encoder_conv1d_3_128_conv1d_expanddims_1_readvariableop_resource:ААH
9tree_encoder_conv1d_3_128_biasadd_readvariableop_resource:	АN
?tree_encoder_batch_norm_7_128_batchnorm_readvariableop_resource:	АR
Ctree_encoder_batch_norm_7_128_batchnorm_mul_readvariableop_resource:	АP
Atree_encoder_batch_norm_7_128_batchnorm_readvariableop_1_resource:	АP
Atree_encoder_batch_norm_7_128_batchnorm_readvariableop_2_resource:	АN
?tree_encoder_batch_norm_5_128_batchnorm_readvariableop_resource:	АR
Ctree_encoder_batch_norm_5_128_batchnorm_mul_readvariableop_resource:	АP
Atree_encoder_batch_norm_5_128_batchnorm_readvariableop_1_resource:	АP
Atree_encoder_batch_norm_5_128_batchnorm_readvariableop_2_resource:	АN
?tree_encoder_batch_norm_3_128_batchnorm_readvariableop_resource:	АR
Ctree_encoder_batch_norm_3_128_batchnorm_mul_readvariableop_resource:	АP
Atree_encoder_batch_norm_3_128_batchnorm_readvariableop_1_resource:	АP
Atree_encoder_batch_norm_3_128_batchnorm_readvariableop_2_resource:	А]
Etree_encoder_conv1d_7_256_conv1d_expanddims_1_readvariableop_resource:ААH
9tree_encoder_conv1d_7_256_biasadd_readvariableop_resource:	А]
Etree_encoder_conv1d_5_256_conv1d_expanddims_1_readvariableop_resource:ААH
9tree_encoder_conv1d_5_256_biasadd_readvariableop_resource:	А]
Etree_encoder_conv1d_3_256_conv1d_expanddims_1_readvariableop_resource:ААH
9tree_encoder_conv1d_3_256_biasadd_readvariableop_resource:	АN
?tree_encoder_batch_norm_7_256_batchnorm_readvariableop_resource:	АR
Ctree_encoder_batch_norm_7_256_batchnorm_mul_readvariableop_resource:	АP
Atree_encoder_batch_norm_7_256_batchnorm_readvariableop_1_resource:	АP
Atree_encoder_batch_norm_7_256_batchnorm_readvariableop_2_resource:	АN
?tree_encoder_batch_norm_5_256_batchnorm_readvariableop_resource:	АR
Ctree_encoder_batch_norm_5_256_batchnorm_mul_readvariableop_resource:	АP
Atree_encoder_batch_norm_5_256_batchnorm_readvariableop_1_resource:	АP
Atree_encoder_batch_norm_5_256_batchnorm_readvariableop_2_resource:	АN
?tree_encoder_batch_norm_3_256_batchnorm_readvariableop_resource:	АR
Ctree_encoder_batch_norm_3_256_batchnorm_mul_readvariableop_resource:	АP
Atree_encoder_batch_norm_3_256_batchnorm_readvariableop_1_resource:	АP
Atree_encoder_batch_norm_3_256_batchnorm_readvariableop_2_resource:	АI
6tree_encoder_prediction_matmul_readvariableop_resource:	АE
7tree_encoder_prediction_biasadd_readvariableop_resource:
identityИв6Tree_Encoder/Batch_Norm_3_128/batchnorm/ReadVariableOpв8Tree_Encoder/Batch_Norm_3_128/batchnorm/ReadVariableOp_1в8Tree_Encoder/Batch_Norm_3_128/batchnorm/ReadVariableOp_2в:Tree_Encoder/Batch_Norm_3_128/batchnorm/mul/ReadVariableOpв6Tree_Encoder/Batch_Norm_3_256/batchnorm/ReadVariableOpв8Tree_Encoder/Batch_Norm_3_256/batchnorm/ReadVariableOp_1в8Tree_Encoder/Batch_Norm_3_256/batchnorm/ReadVariableOp_2в:Tree_Encoder/Batch_Norm_3_256/batchnorm/mul/ReadVariableOpв6Tree_Encoder/Batch_Norm_5_128/batchnorm/ReadVariableOpв8Tree_Encoder/Batch_Norm_5_128/batchnorm/ReadVariableOp_1в8Tree_Encoder/Batch_Norm_5_128/batchnorm/ReadVariableOp_2в:Tree_Encoder/Batch_Norm_5_128/batchnorm/mul/ReadVariableOpв6Tree_Encoder/Batch_Norm_5_256/batchnorm/ReadVariableOpв8Tree_Encoder/Batch_Norm_5_256/batchnorm/ReadVariableOp_1в8Tree_Encoder/Batch_Norm_5_256/batchnorm/ReadVariableOp_2в:Tree_Encoder/Batch_Norm_5_256/batchnorm/mul/ReadVariableOpв6Tree_Encoder/Batch_Norm_7_128/batchnorm/ReadVariableOpв8Tree_Encoder/Batch_Norm_7_128/batchnorm/ReadVariableOp_1в8Tree_Encoder/Batch_Norm_7_128/batchnorm/ReadVariableOp_2в:Tree_Encoder/Batch_Norm_7_128/batchnorm/mul/ReadVariableOpв6Tree_Encoder/Batch_Norm_7_256/batchnorm/ReadVariableOpв8Tree_Encoder/Batch_Norm_7_256/batchnorm/ReadVariableOp_1в8Tree_Encoder/Batch_Norm_7_256/batchnorm/ReadVariableOp_2в:Tree_Encoder/Batch_Norm_7_256/batchnorm/mul/ReadVariableOpв0Tree_Encoder/Conv1D_3_128/BiasAdd/ReadVariableOpв<Tree_Encoder/Conv1D_3_128/conv1d/ExpandDims_1/ReadVariableOpв0Tree_Encoder/Conv1D_3_256/BiasAdd/ReadVariableOpв<Tree_Encoder/Conv1D_3_256/conv1d/ExpandDims_1/ReadVariableOpв0Tree_Encoder/Conv1D_5_128/BiasAdd/ReadVariableOpв<Tree_Encoder/Conv1D_5_128/conv1d/ExpandDims_1/ReadVariableOpв0Tree_Encoder/Conv1D_5_256/BiasAdd/ReadVariableOpв<Tree_Encoder/Conv1D_5_256/conv1d/ExpandDims_1/ReadVariableOpв0Tree_Encoder/Conv1D_7_128/BiasAdd/ReadVariableOpв<Tree_Encoder/Conv1D_7_128/conv1d/ExpandDims_1/ReadVariableOpв0Tree_Encoder/Conv1D_7_256/BiasAdd/ReadVariableOpв<Tree_Encoder/Conv1D_7_256/conv1d/ExpandDims_1/ReadVariableOpв.Tree_Encoder/Prediction/BiasAdd/ReadVariableOpв-Tree_Encoder/Prediction/MatMul/ReadVariableOpж
'Tree_Encoder/Embedding_Dropout/IdentityIdentityinputs*
T0*5
_output_shapes#
!:                  А2)
'Tree_Encoder/Embedding_Dropout/Identityн
/Tree_Encoder/Conv1D_7_128/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        21
/Tree_Encoder/Conv1D_7_128/conv1d/ExpandDims/dimШ
+Tree_Encoder/Conv1D_7_128/conv1d/ExpandDims
ExpandDims0Tree_Encoder/Embedding_Dropout/Identity:output:08Tree_Encoder/Conv1D_7_128/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#                  А2-
+Tree_Encoder/Conv1D_7_128/conv1d/ExpandDimsИ
<Tree_Encoder/Conv1D_7_128/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEtree_encoder_conv1d_7_128_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:АА*
dtype02>
<Tree_Encoder/Conv1D_7_128/conv1d/ExpandDims_1/ReadVariableOpи
1Tree_Encoder/Conv1D_7_128/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1Tree_Encoder/Conv1D_7_128/conv1d/ExpandDims_1/dimб
-Tree_Encoder/Conv1D_7_128/conv1d/ExpandDims_1
ExpandDimsDTree_Encoder/Conv1D_7_128/conv1d/ExpandDims_1/ReadVariableOp:value:0:Tree_Encoder/Conv1D_7_128/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:АА2/
-Tree_Encoder/Conv1D_7_128/conv1d/ExpandDims_1и
 Tree_Encoder/Conv1D_7_128/conv1dConv2D4Tree_Encoder/Conv1D_7_128/conv1d/ExpandDims:output:06Tree_Encoder/Conv1D_7_128/conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#                  А*
paddingSAME*
strides
2"
 Tree_Encoder/Conv1D_7_128/conv1dъ
(Tree_Encoder/Conv1D_7_128/conv1d/SqueezeSqueeze)Tree_Encoder/Conv1D_7_128/conv1d:output:0*
T0*5
_output_shapes#
!:                  А*
squeeze_dims

¤        2*
(Tree_Encoder/Conv1D_7_128/conv1d/Squeeze█
0Tree_Encoder/Conv1D_7_128/BiasAdd/ReadVariableOpReadVariableOp9tree_encoder_conv1d_7_128_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype022
0Tree_Encoder/Conv1D_7_128/BiasAdd/ReadVariableOp■
!Tree_Encoder/Conv1D_7_128/BiasAddBiasAdd1Tree_Encoder/Conv1D_7_128/conv1d/Squeeze:output:08Tree_Encoder/Conv1D_7_128/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:                  А2#
!Tree_Encoder/Conv1D_7_128/BiasAdd╜
!Tree_Encoder/Conv1D_7_128/SigmoidSigmoid*Tree_Encoder/Conv1D_7_128/BiasAdd:output:0*
T0*5
_output_shapes#
!:                  А2#
!Tree_Encoder/Conv1D_7_128/Sigmoid╪
Tree_Encoder/Conv1D_7_128/mulMul*Tree_Encoder/Conv1D_7_128/BiasAdd:output:0%Tree_Encoder/Conv1D_7_128/Sigmoid:y:0*
T0*5
_output_shapes#
!:                  А2
Tree_Encoder/Conv1D_7_128/mul╖
"Tree_Encoder/Conv1D_7_128/IdentityIdentity!Tree_Encoder/Conv1D_7_128/mul:z:0*
T0*5
_output_shapes#
!:                  А2$
"Tree_Encoder/Conv1D_7_128/Identity╜
#Tree_Encoder/Conv1D_7_128/IdentityN	IdentityN!Tree_Encoder/Conv1D_7_128/mul:z:0*Tree_Encoder/Conv1D_7_128/BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1294347631*V
_output_shapesD
B:                  А:                  А2%
#Tree_Encoder/Conv1D_7_128/IdentityNн
/Tree_Encoder/Conv1D_5_128/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        21
/Tree_Encoder/Conv1D_5_128/conv1d/ExpandDims/dimШ
+Tree_Encoder/Conv1D_5_128/conv1d/ExpandDims
ExpandDims0Tree_Encoder/Embedding_Dropout/Identity:output:08Tree_Encoder/Conv1D_5_128/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#                  А2-
+Tree_Encoder/Conv1D_5_128/conv1d/ExpandDimsИ
<Tree_Encoder/Conv1D_5_128/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEtree_encoder_conv1d_5_128_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:АА*
dtype02>
<Tree_Encoder/Conv1D_5_128/conv1d/ExpandDims_1/ReadVariableOpи
1Tree_Encoder/Conv1D_5_128/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1Tree_Encoder/Conv1D_5_128/conv1d/ExpandDims_1/dimб
-Tree_Encoder/Conv1D_5_128/conv1d/ExpandDims_1
ExpandDimsDTree_Encoder/Conv1D_5_128/conv1d/ExpandDims_1/ReadVariableOp:value:0:Tree_Encoder/Conv1D_5_128/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:АА2/
-Tree_Encoder/Conv1D_5_128/conv1d/ExpandDims_1и
 Tree_Encoder/Conv1D_5_128/conv1dConv2D4Tree_Encoder/Conv1D_5_128/conv1d/ExpandDims:output:06Tree_Encoder/Conv1D_5_128/conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#                  А*
paddingSAME*
strides
2"
 Tree_Encoder/Conv1D_5_128/conv1dъ
(Tree_Encoder/Conv1D_5_128/conv1d/SqueezeSqueeze)Tree_Encoder/Conv1D_5_128/conv1d:output:0*
T0*5
_output_shapes#
!:                  А*
squeeze_dims

¤        2*
(Tree_Encoder/Conv1D_5_128/conv1d/Squeeze█
0Tree_Encoder/Conv1D_5_128/BiasAdd/ReadVariableOpReadVariableOp9tree_encoder_conv1d_5_128_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype022
0Tree_Encoder/Conv1D_5_128/BiasAdd/ReadVariableOp■
!Tree_Encoder/Conv1D_5_128/BiasAddBiasAdd1Tree_Encoder/Conv1D_5_128/conv1d/Squeeze:output:08Tree_Encoder/Conv1D_5_128/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:                  А2#
!Tree_Encoder/Conv1D_5_128/BiasAdd╜
!Tree_Encoder/Conv1D_5_128/SigmoidSigmoid*Tree_Encoder/Conv1D_5_128/BiasAdd:output:0*
T0*5
_output_shapes#
!:                  А2#
!Tree_Encoder/Conv1D_5_128/Sigmoid╪
Tree_Encoder/Conv1D_5_128/mulMul*Tree_Encoder/Conv1D_5_128/BiasAdd:output:0%Tree_Encoder/Conv1D_5_128/Sigmoid:y:0*
T0*5
_output_shapes#
!:                  А2
Tree_Encoder/Conv1D_5_128/mul╖
"Tree_Encoder/Conv1D_5_128/IdentityIdentity!Tree_Encoder/Conv1D_5_128/mul:z:0*
T0*5
_output_shapes#
!:                  А2$
"Tree_Encoder/Conv1D_5_128/Identity╜
#Tree_Encoder/Conv1D_5_128/IdentityN	IdentityN!Tree_Encoder/Conv1D_5_128/mul:z:0*Tree_Encoder/Conv1D_5_128/BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1294347648*V
_output_shapesD
B:                  А:                  А2%
#Tree_Encoder/Conv1D_5_128/IdentityNн
/Tree_Encoder/Conv1D_3_128/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        21
/Tree_Encoder/Conv1D_3_128/conv1d/ExpandDims/dimШ
+Tree_Encoder/Conv1D_3_128/conv1d/ExpandDims
ExpandDims0Tree_Encoder/Embedding_Dropout/Identity:output:08Tree_Encoder/Conv1D_3_128/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#                  А2-
+Tree_Encoder/Conv1D_3_128/conv1d/ExpandDimsИ
<Tree_Encoder/Conv1D_3_128/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEtree_encoder_conv1d_3_128_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:АА*
dtype02>
<Tree_Encoder/Conv1D_3_128/conv1d/ExpandDims_1/ReadVariableOpи
1Tree_Encoder/Conv1D_3_128/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1Tree_Encoder/Conv1D_3_128/conv1d/ExpandDims_1/dimб
-Tree_Encoder/Conv1D_3_128/conv1d/ExpandDims_1
ExpandDimsDTree_Encoder/Conv1D_3_128/conv1d/ExpandDims_1/ReadVariableOp:value:0:Tree_Encoder/Conv1D_3_128/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:АА2/
-Tree_Encoder/Conv1D_3_128/conv1d/ExpandDims_1и
 Tree_Encoder/Conv1D_3_128/conv1dConv2D4Tree_Encoder/Conv1D_3_128/conv1d/ExpandDims:output:06Tree_Encoder/Conv1D_3_128/conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#                  А*
paddingSAME*
strides
2"
 Tree_Encoder/Conv1D_3_128/conv1dъ
(Tree_Encoder/Conv1D_3_128/conv1d/SqueezeSqueeze)Tree_Encoder/Conv1D_3_128/conv1d:output:0*
T0*5
_output_shapes#
!:                  А*
squeeze_dims

¤        2*
(Tree_Encoder/Conv1D_3_128/conv1d/Squeeze█
0Tree_Encoder/Conv1D_3_128/BiasAdd/ReadVariableOpReadVariableOp9tree_encoder_conv1d_3_128_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype022
0Tree_Encoder/Conv1D_3_128/BiasAdd/ReadVariableOp■
!Tree_Encoder/Conv1D_3_128/BiasAddBiasAdd1Tree_Encoder/Conv1D_3_128/conv1d/Squeeze:output:08Tree_Encoder/Conv1D_3_128/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:                  А2#
!Tree_Encoder/Conv1D_3_128/BiasAdd╜
!Tree_Encoder/Conv1D_3_128/SigmoidSigmoid*Tree_Encoder/Conv1D_3_128/BiasAdd:output:0*
T0*5
_output_shapes#
!:                  А2#
!Tree_Encoder/Conv1D_3_128/Sigmoid╪
Tree_Encoder/Conv1D_3_128/mulMul*Tree_Encoder/Conv1D_3_128/BiasAdd:output:0%Tree_Encoder/Conv1D_3_128/Sigmoid:y:0*
T0*5
_output_shapes#
!:                  А2
Tree_Encoder/Conv1D_3_128/mul╖
"Tree_Encoder/Conv1D_3_128/IdentityIdentity!Tree_Encoder/Conv1D_3_128/mul:z:0*
T0*5
_output_shapes#
!:                  А2$
"Tree_Encoder/Conv1D_3_128/Identity╜
#Tree_Encoder/Conv1D_3_128/IdentityN	IdentityN!Tree_Encoder/Conv1D_3_128/mul:z:0*Tree_Encoder/Conv1D_3_128/BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1294347665*V
_output_shapesD
B:                  А:                  А2%
#Tree_Encoder/Conv1D_3_128/IdentityNэ
6Tree_Encoder/Batch_Norm_7_128/batchnorm/ReadVariableOpReadVariableOp?tree_encoder_batch_norm_7_128_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype028
6Tree_Encoder/Batch_Norm_7_128/batchnorm/ReadVariableOpг
-Tree_Encoder/Batch_Norm_7_128/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2/
-Tree_Encoder/Batch_Norm_7_128/batchnorm/add/yБ
+Tree_Encoder/Batch_Norm_7_128/batchnorm/addAddV2>Tree_Encoder/Batch_Norm_7_128/batchnorm/ReadVariableOp:value:06Tree_Encoder/Batch_Norm_7_128/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2-
+Tree_Encoder/Batch_Norm_7_128/batchnorm/add╛
-Tree_Encoder/Batch_Norm_7_128/batchnorm/RsqrtRsqrt/Tree_Encoder/Batch_Norm_7_128/batchnorm/add:z:0*
T0*
_output_shapes	
:А2/
-Tree_Encoder/Batch_Norm_7_128/batchnorm/Rsqrt∙
:Tree_Encoder/Batch_Norm_7_128/batchnorm/mul/ReadVariableOpReadVariableOpCtree_encoder_batch_norm_7_128_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02<
:Tree_Encoder/Batch_Norm_7_128/batchnorm/mul/ReadVariableOp■
+Tree_Encoder/Batch_Norm_7_128/batchnorm/mulMul1Tree_Encoder/Batch_Norm_7_128/batchnorm/Rsqrt:y:0BTree_Encoder/Batch_Norm_7_128/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2-
+Tree_Encoder/Batch_Norm_7_128/batchnorm/mulД
-Tree_Encoder/Batch_Norm_7_128/batchnorm/mul_1Mul,Tree_Encoder/Conv1D_7_128/IdentityN:output:0/Tree_Encoder/Batch_Norm_7_128/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2/
-Tree_Encoder/Batch_Norm_7_128/batchnorm/mul_1є
8Tree_Encoder/Batch_Norm_7_128/batchnorm/ReadVariableOp_1ReadVariableOpAtree_encoder_batch_norm_7_128_batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02:
8Tree_Encoder/Batch_Norm_7_128/batchnorm/ReadVariableOp_1■
-Tree_Encoder/Batch_Norm_7_128/batchnorm/mul_2Mul@Tree_Encoder/Batch_Norm_7_128/batchnorm/ReadVariableOp_1:value:0/Tree_Encoder/Batch_Norm_7_128/batchnorm/mul:z:0*
T0*
_output_shapes	
:А2/
-Tree_Encoder/Batch_Norm_7_128/batchnorm/mul_2є
8Tree_Encoder/Batch_Norm_7_128/batchnorm/ReadVariableOp_2ReadVariableOpAtree_encoder_batch_norm_7_128_batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02:
8Tree_Encoder/Batch_Norm_7_128/batchnorm/ReadVariableOp_2№
+Tree_Encoder/Batch_Norm_7_128/batchnorm/subSub@Tree_Encoder/Batch_Norm_7_128/batchnorm/ReadVariableOp_2:value:01Tree_Encoder/Batch_Norm_7_128/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2-
+Tree_Encoder/Batch_Norm_7_128/batchnorm/subЛ
-Tree_Encoder/Batch_Norm_7_128/batchnorm/add_1AddV21Tree_Encoder/Batch_Norm_7_128/batchnorm/mul_1:z:0/Tree_Encoder/Batch_Norm_7_128/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2/
-Tree_Encoder/Batch_Norm_7_128/batchnorm/add_1э
6Tree_Encoder/Batch_Norm_5_128/batchnorm/ReadVariableOpReadVariableOp?tree_encoder_batch_norm_5_128_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype028
6Tree_Encoder/Batch_Norm_5_128/batchnorm/ReadVariableOpг
-Tree_Encoder/Batch_Norm_5_128/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2/
-Tree_Encoder/Batch_Norm_5_128/batchnorm/add/yБ
+Tree_Encoder/Batch_Norm_5_128/batchnorm/addAddV2>Tree_Encoder/Batch_Norm_5_128/batchnorm/ReadVariableOp:value:06Tree_Encoder/Batch_Norm_5_128/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2-
+Tree_Encoder/Batch_Norm_5_128/batchnorm/add╛
-Tree_Encoder/Batch_Norm_5_128/batchnorm/RsqrtRsqrt/Tree_Encoder/Batch_Norm_5_128/batchnorm/add:z:0*
T0*
_output_shapes	
:А2/
-Tree_Encoder/Batch_Norm_5_128/batchnorm/Rsqrt∙
:Tree_Encoder/Batch_Norm_5_128/batchnorm/mul/ReadVariableOpReadVariableOpCtree_encoder_batch_norm_5_128_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02<
:Tree_Encoder/Batch_Norm_5_128/batchnorm/mul/ReadVariableOp■
+Tree_Encoder/Batch_Norm_5_128/batchnorm/mulMul1Tree_Encoder/Batch_Norm_5_128/batchnorm/Rsqrt:y:0BTree_Encoder/Batch_Norm_5_128/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2-
+Tree_Encoder/Batch_Norm_5_128/batchnorm/mulД
-Tree_Encoder/Batch_Norm_5_128/batchnorm/mul_1Mul,Tree_Encoder/Conv1D_5_128/IdentityN:output:0/Tree_Encoder/Batch_Norm_5_128/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2/
-Tree_Encoder/Batch_Norm_5_128/batchnorm/mul_1є
8Tree_Encoder/Batch_Norm_5_128/batchnorm/ReadVariableOp_1ReadVariableOpAtree_encoder_batch_norm_5_128_batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02:
8Tree_Encoder/Batch_Norm_5_128/batchnorm/ReadVariableOp_1■
-Tree_Encoder/Batch_Norm_5_128/batchnorm/mul_2Mul@Tree_Encoder/Batch_Norm_5_128/batchnorm/ReadVariableOp_1:value:0/Tree_Encoder/Batch_Norm_5_128/batchnorm/mul:z:0*
T0*
_output_shapes	
:А2/
-Tree_Encoder/Batch_Norm_5_128/batchnorm/mul_2є
8Tree_Encoder/Batch_Norm_5_128/batchnorm/ReadVariableOp_2ReadVariableOpAtree_encoder_batch_norm_5_128_batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02:
8Tree_Encoder/Batch_Norm_5_128/batchnorm/ReadVariableOp_2№
+Tree_Encoder/Batch_Norm_5_128/batchnorm/subSub@Tree_Encoder/Batch_Norm_5_128/batchnorm/ReadVariableOp_2:value:01Tree_Encoder/Batch_Norm_5_128/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2-
+Tree_Encoder/Batch_Norm_5_128/batchnorm/subЛ
-Tree_Encoder/Batch_Norm_5_128/batchnorm/add_1AddV21Tree_Encoder/Batch_Norm_5_128/batchnorm/mul_1:z:0/Tree_Encoder/Batch_Norm_5_128/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2/
-Tree_Encoder/Batch_Norm_5_128/batchnorm/add_1э
6Tree_Encoder/Batch_Norm_3_128/batchnorm/ReadVariableOpReadVariableOp?tree_encoder_batch_norm_3_128_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype028
6Tree_Encoder/Batch_Norm_3_128/batchnorm/ReadVariableOpг
-Tree_Encoder/Batch_Norm_3_128/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2/
-Tree_Encoder/Batch_Norm_3_128/batchnorm/add/yБ
+Tree_Encoder/Batch_Norm_3_128/batchnorm/addAddV2>Tree_Encoder/Batch_Norm_3_128/batchnorm/ReadVariableOp:value:06Tree_Encoder/Batch_Norm_3_128/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2-
+Tree_Encoder/Batch_Norm_3_128/batchnorm/add╛
-Tree_Encoder/Batch_Norm_3_128/batchnorm/RsqrtRsqrt/Tree_Encoder/Batch_Norm_3_128/batchnorm/add:z:0*
T0*
_output_shapes	
:А2/
-Tree_Encoder/Batch_Norm_3_128/batchnorm/Rsqrt∙
:Tree_Encoder/Batch_Norm_3_128/batchnorm/mul/ReadVariableOpReadVariableOpCtree_encoder_batch_norm_3_128_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02<
:Tree_Encoder/Batch_Norm_3_128/batchnorm/mul/ReadVariableOp■
+Tree_Encoder/Batch_Norm_3_128/batchnorm/mulMul1Tree_Encoder/Batch_Norm_3_128/batchnorm/Rsqrt:y:0BTree_Encoder/Batch_Norm_3_128/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2-
+Tree_Encoder/Batch_Norm_3_128/batchnorm/mulД
-Tree_Encoder/Batch_Norm_3_128/batchnorm/mul_1Mul,Tree_Encoder/Conv1D_3_128/IdentityN:output:0/Tree_Encoder/Batch_Norm_3_128/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2/
-Tree_Encoder/Batch_Norm_3_128/batchnorm/mul_1є
8Tree_Encoder/Batch_Norm_3_128/batchnorm/ReadVariableOp_1ReadVariableOpAtree_encoder_batch_norm_3_128_batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02:
8Tree_Encoder/Batch_Norm_3_128/batchnorm/ReadVariableOp_1■
-Tree_Encoder/Batch_Norm_3_128/batchnorm/mul_2Mul@Tree_Encoder/Batch_Norm_3_128/batchnorm/ReadVariableOp_1:value:0/Tree_Encoder/Batch_Norm_3_128/batchnorm/mul:z:0*
T0*
_output_shapes	
:А2/
-Tree_Encoder/Batch_Norm_3_128/batchnorm/mul_2є
8Tree_Encoder/Batch_Norm_3_128/batchnorm/ReadVariableOp_2ReadVariableOpAtree_encoder_batch_norm_3_128_batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02:
8Tree_Encoder/Batch_Norm_3_128/batchnorm/ReadVariableOp_2№
+Tree_Encoder/Batch_Norm_3_128/batchnorm/subSub@Tree_Encoder/Batch_Norm_3_128/batchnorm/ReadVariableOp_2:value:01Tree_Encoder/Batch_Norm_3_128/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2-
+Tree_Encoder/Batch_Norm_3_128/batchnorm/subЛ
-Tree_Encoder/Batch_Norm_3_128/batchnorm/add_1AddV21Tree_Encoder/Batch_Norm_3_128/batchnorm/mul_1:z:0/Tree_Encoder/Batch_Norm_3_128/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2/
-Tree_Encoder/Batch_Norm_3_128/batchnorm/add_1н
/Tree_Encoder/Conv1D_7_256/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        21
/Tree_Encoder/Conv1D_7_256/conv1d/ExpandDims/dimЩ
+Tree_Encoder/Conv1D_7_256/conv1d/ExpandDims
ExpandDims1Tree_Encoder/Batch_Norm_7_128/batchnorm/add_1:z:08Tree_Encoder/Conv1D_7_256/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#                  А2-
+Tree_Encoder/Conv1D_7_256/conv1d/ExpandDimsИ
<Tree_Encoder/Conv1D_7_256/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEtree_encoder_conv1d_7_256_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:АА*
dtype02>
<Tree_Encoder/Conv1D_7_256/conv1d/ExpandDims_1/ReadVariableOpи
1Tree_Encoder/Conv1D_7_256/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1Tree_Encoder/Conv1D_7_256/conv1d/ExpandDims_1/dimб
-Tree_Encoder/Conv1D_7_256/conv1d/ExpandDims_1
ExpandDimsDTree_Encoder/Conv1D_7_256/conv1d/ExpandDims_1/ReadVariableOp:value:0:Tree_Encoder/Conv1D_7_256/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:АА2/
-Tree_Encoder/Conv1D_7_256/conv1d/ExpandDims_1и
 Tree_Encoder/Conv1D_7_256/conv1dConv2D4Tree_Encoder/Conv1D_7_256/conv1d/ExpandDims:output:06Tree_Encoder/Conv1D_7_256/conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#                  А*
paddingSAME*
strides
2"
 Tree_Encoder/Conv1D_7_256/conv1dъ
(Tree_Encoder/Conv1D_7_256/conv1d/SqueezeSqueeze)Tree_Encoder/Conv1D_7_256/conv1d:output:0*
T0*5
_output_shapes#
!:                  А*
squeeze_dims

¤        2*
(Tree_Encoder/Conv1D_7_256/conv1d/Squeeze█
0Tree_Encoder/Conv1D_7_256/BiasAdd/ReadVariableOpReadVariableOp9tree_encoder_conv1d_7_256_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype022
0Tree_Encoder/Conv1D_7_256/BiasAdd/ReadVariableOp■
!Tree_Encoder/Conv1D_7_256/BiasAddBiasAdd1Tree_Encoder/Conv1D_7_256/conv1d/Squeeze:output:08Tree_Encoder/Conv1D_7_256/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:                  А2#
!Tree_Encoder/Conv1D_7_256/BiasAdd╜
!Tree_Encoder/Conv1D_7_256/SigmoidSigmoid*Tree_Encoder/Conv1D_7_256/BiasAdd:output:0*
T0*5
_output_shapes#
!:                  А2#
!Tree_Encoder/Conv1D_7_256/Sigmoid╪
Tree_Encoder/Conv1D_7_256/mulMul*Tree_Encoder/Conv1D_7_256/BiasAdd:output:0%Tree_Encoder/Conv1D_7_256/Sigmoid:y:0*
T0*5
_output_shapes#
!:                  А2
Tree_Encoder/Conv1D_7_256/mul╖
"Tree_Encoder/Conv1D_7_256/IdentityIdentity!Tree_Encoder/Conv1D_7_256/mul:z:0*
T0*5
_output_shapes#
!:                  А2$
"Tree_Encoder/Conv1D_7_256/Identity╜
#Tree_Encoder/Conv1D_7_256/IdentityN	IdentityN!Tree_Encoder/Conv1D_7_256/mul:z:0*Tree_Encoder/Conv1D_7_256/BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1294347730*V
_output_shapesD
B:                  А:                  А2%
#Tree_Encoder/Conv1D_7_256/IdentityNн
/Tree_Encoder/Conv1D_5_256/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        21
/Tree_Encoder/Conv1D_5_256/conv1d/ExpandDims/dimЩ
+Tree_Encoder/Conv1D_5_256/conv1d/ExpandDims
ExpandDims1Tree_Encoder/Batch_Norm_5_128/batchnorm/add_1:z:08Tree_Encoder/Conv1D_5_256/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#                  А2-
+Tree_Encoder/Conv1D_5_256/conv1d/ExpandDimsИ
<Tree_Encoder/Conv1D_5_256/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEtree_encoder_conv1d_5_256_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:АА*
dtype02>
<Tree_Encoder/Conv1D_5_256/conv1d/ExpandDims_1/ReadVariableOpи
1Tree_Encoder/Conv1D_5_256/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1Tree_Encoder/Conv1D_5_256/conv1d/ExpandDims_1/dimб
-Tree_Encoder/Conv1D_5_256/conv1d/ExpandDims_1
ExpandDimsDTree_Encoder/Conv1D_5_256/conv1d/ExpandDims_1/ReadVariableOp:value:0:Tree_Encoder/Conv1D_5_256/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:АА2/
-Tree_Encoder/Conv1D_5_256/conv1d/ExpandDims_1и
 Tree_Encoder/Conv1D_5_256/conv1dConv2D4Tree_Encoder/Conv1D_5_256/conv1d/ExpandDims:output:06Tree_Encoder/Conv1D_5_256/conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#                  А*
paddingSAME*
strides
2"
 Tree_Encoder/Conv1D_5_256/conv1dъ
(Tree_Encoder/Conv1D_5_256/conv1d/SqueezeSqueeze)Tree_Encoder/Conv1D_5_256/conv1d:output:0*
T0*5
_output_shapes#
!:                  А*
squeeze_dims

¤        2*
(Tree_Encoder/Conv1D_5_256/conv1d/Squeeze█
0Tree_Encoder/Conv1D_5_256/BiasAdd/ReadVariableOpReadVariableOp9tree_encoder_conv1d_5_256_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype022
0Tree_Encoder/Conv1D_5_256/BiasAdd/ReadVariableOp■
!Tree_Encoder/Conv1D_5_256/BiasAddBiasAdd1Tree_Encoder/Conv1D_5_256/conv1d/Squeeze:output:08Tree_Encoder/Conv1D_5_256/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:                  А2#
!Tree_Encoder/Conv1D_5_256/BiasAdd╜
!Tree_Encoder/Conv1D_5_256/SigmoidSigmoid*Tree_Encoder/Conv1D_5_256/BiasAdd:output:0*
T0*5
_output_shapes#
!:                  А2#
!Tree_Encoder/Conv1D_5_256/Sigmoid╪
Tree_Encoder/Conv1D_5_256/mulMul*Tree_Encoder/Conv1D_5_256/BiasAdd:output:0%Tree_Encoder/Conv1D_5_256/Sigmoid:y:0*
T0*5
_output_shapes#
!:                  А2
Tree_Encoder/Conv1D_5_256/mul╖
"Tree_Encoder/Conv1D_5_256/IdentityIdentity!Tree_Encoder/Conv1D_5_256/mul:z:0*
T0*5
_output_shapes#
!:                  А2$
"Tree_Encoder/Conv1D_5_256/Identity╜
#Tree_Encoder/Conv1D_5_256/IdentityN	IdentityN!Tree_Encoder/Conv1D_5_256/mul:z:0*Tree_Encoder/Conv1D_5_256/BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1294347747*V
_output_shapesD
B:                  А:                  А2%
#Tree_Encoder/Conv1D_5_256/IdentityNн
/Tree_Encoder/Conv1D_3_256/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        21
/Tree_Encoder/Conv1D_3_256/conv1d/ExpandDims/dimЩ
+Tree_Encoder/Conv1D_3_256/conv1d/ExpandDims
ExpandDims1Tree_Encoder/Batch_Norm_3_128/batchnorm/add_1:z:08Tree_Encoder/Conv1D_3_256/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#                  А2-
+Tree_Encoder/Conv1D_3_256/conv1d/ExpandDimsИ
<Tree_Encoder/Conv1D_3_256/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEtree_encoder_conv1d_3_256_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:АА*
dtype02>
<Tree_Encoder/Conv1D_3_256/conv1d/ExpandDims_1/ReadVariableOpи
1Tree_Encoder/Conv1D_3_256/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1Tree_Encoder/Conv1D_3_256/conv1d/ExpandDims_1/dimб
-Tree_Encoder/Conv1D_3_256/conv1d/ExpandDims_1
ExpandDimsDTree_Encoder/Conv1D_3_256/conv1d/ExpandDims_1/ReadVariableOp:value:0:Tree_Encoder/Conv1D_3_256/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:АА2/
-Tree_Encoder/Conv1D_3_256/conv1d/ExpandDims_1и
 Tree_Encoder/Conv1D_3_256/conv1dConv2D4Tree_Encoder/Conv1D_3_256/conv1d/ExpandDims:output:06Tree_Encoder/Conv1D_3_256/conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#                  А*
paddingSAME*
strides
2"
 Tree_Encoder/Conv1D_3_256/conv1dъ
(Tree_Encoder/Conv1D_3_256/conv1d/SqueezeSqueeze)Tree_Encoder/Conv1D_3_256/conv1d:output:0*
T0*5
_output_shapes#
!:                  А*
squeeze_dims

¤        2*
(Tree_Encoder/Conv1D_3_256/conv1d/Squeeze█
0Tree_Encoder/Conv1D_3_256/BiasAdd/ReadVariableOpReadVariableOp9tree_encoder_conv1d_3_256_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype022
0Tree_Encoder/Conv1D_3_256/BiasAdd/ReadVariableOp■
!Tree_Encoder/Conv1D_3_256/BiasAddBiasAdd1Tree_Encoder/Conv1D_3_256/conv1d/Squeeze:output:08Tree_Encoder/Conv1D_3_256/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:                  А2#
!Tree_Encoder/Conv1D_3_256/BiasAdd╜
!Tree_Encoder/Conv1D_3_256/SigmoidSigmoid*Tree_Encoder/Conv1D_3_256/BiasAdd:output:0*
T0*5
_output_shapes#
!:                  А2#
!Tree_Encoder/Conv1D_3_256/Sigmoid╪
Tree_Encoder/Conv1D_3_256/mulMul*Tree_Encoder/Conv1D_3_256/BiasAdd:output:0%Tree_Encoder/Conv1D_3_256/Sigmoid:y:0*
T0*5
_output_shapes#
!:                  А2
Tree_Encoder/Conv1D_3_256/mul╖
"Tree_Encoder/Conv1D_3_256/IdentityIdentity!Tree_Encoder/Conv1D_3_256/mul:z:0*
T0*5
_output_shapes#
!:                  А2$
"Tree_Encoder/Conv1D_3_256/Identity╜
#Tree_Encoder/Conv1D_3_256/IdentityN	IdentityN!Tree_Encoder/Conv1D_3_256/mul:z:0*Tree_Encoder/Conv1D_3_256/BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1294347764*V
_output_shapesD
B:                  А:                  А2%
#Tree_Encoder/Conv1D_3_256/IdentityNэ
6Tree_Encoder/Batch_Norm_7_256/batchnorm/ReadVariableOpReadVariableOp?tree_encoder_batch_norm_7_256_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype028
6Tree_Encoder/Batch_Norm_7_256/batchnorm/ReadVariableOpг
-Tree_Encoder/Batch_Norm_7_256/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2/
-Tree_Encoder/Batch_Norm_7_256/batchnorm/add/yБ
+Tree_Encoder/Batch_Norm_7_256/batchnorm/addAddV2>Tree_Encoder/Batch_Norm_7_256/batchnorm/ReadVariableOp:value:06Tree_Encoder/Batch_Norm_7_256/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2-
+Tree_Encoder/Batch_Norm_7_256/batchnorm/add╛
-Tree_Encoder/Batch_Norm_7_256/batchnorm/RsqrtRsqrt/Tree_Encoder/Batch_Norm_7_256/batchnorm/add:z:0*
T0*
_output_shapes	
:А2/
-Tree_Encoder/Batch_Norm_7_256/batchnorm/Rsqrt∙
:Tree_Encoder/Batch_Norm_7_256/batchnorm/mul/ReadVariableOpReadVariableOpCtree_encoder_batch_norm_7_256_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02<
:Tree_Encoder/Batch_Norm_7_256/batchnorm/mul/ReadVariableOp■
+Tree_Encoder/Batch_Norm_7_256/batchnorm/mulMul1Tree_Encoder/Batch_Norm_7_256/batchnorm/Rsqrt:y:0BTree_Encoder/Batch_Norm_7_256/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2-
+Tree_Encoder/Batch_Norm_7_256/batchnorm/mulД
-Tree_Encoder/Batch_Norm_7_256/batchnorm/mul_1Mul,Tree_Encoder/Conv1D_7_256/IdentityN:output:0/Tree_Encoder/Batch_Norm_7_256/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2/
-Tree_Encoder/Batch_Norm_7_256/batchnorm/mul_1є
8Tree_Encoder/Batch_Norm_7_256/batchnorm/ReadVariableOp_1ReadVariableOpAtree_encoder_batch_norm_7_256_batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02:
8Tree_Encoder/Batch_Norm_7_256/batchnorm/ReadVariableOp_1■
-Tree_Encoder/Batch_Norm_7_256/batchnorm/mul_2Mul@Tree_Encoder/Batch_Norm_7_256/batchnorm/ReadVariableOp_1:value:0/Tree_Encoder/Batch_Norm_7_256/batchnorm/mul:z:0*
T0*
_output_shapes	
:А2/
-Tree_Encoder/Batch_Norm_7_256/batchnorm/mul_2є
8Tree_Encoder/Batch_Norm_7_256/batchnorm/ReadVariableOp_2ReadVariableOpAtree_encoder_batch_norm_7_256_batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02:
8Tree_Encoder/Batch_Norm_7_256/batchnorm/ReadVariableOp_2№
+Tree_Encoder/Batch_Norm_7_256/batchnorm/subSub@Tree_Encoder/Batch_Norm_7_256/batchnorm/ReadVariableOp_2:value:01Tree_Encoder/Batch_Norm_7_256/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2-
+Tree_Encoder/Batch_Norm_7_256/batchnorm/subЛ
-Tree_Encoder/Batch_Norm_7_256/batchnorm/add_1AddV21Tree_Encoder/Batch_Norm_7_256/batchnorm/mul_1:z:0/Tree_Encoder/Batch_Norm_7_256/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2/
-Tree_Encoder/Batch_Norm_7_256/batchnorm/add_1э
6Tree_Encoder/Batch_Norm_5_256/batchnorm/ReadVariableOpReadVariableOp?tree_encoder_batch_norm_5_256_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype028
6Tree_Encoder/Batch_Norm_5_256/batchnorm/ReadVariableOpг
-Tree_Encoder/Batch_Norm_5_256/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2/
-Tree_Encoder/Batch_Norm_5_256/batchnorm/add/yБ
+Tree_Encoder/Batch_Norm_5_256/batchnorm/addAddV2>Tree_Encoder/Batch_Norm_5_256/batchnorm/ReadVariableOp:value:06Tree_Encoder/Batch_Norm_5_256/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2-
+Tree_Encoder/Batch_Norm_5_256/batchnorm/add╛
-Tree_Encoder/Batch_Norm_5_256/batchnorm/RsqrtRsqrt/Tree_Encoder/Batch_Norm_5_256/batchnorm/add:z:0*
T0*
_output_shapes	
:А2/
-Tree_Encoder/Batch_Norm_5_256/batchnorm/Rsqrt∙
:Tree_Encoder/Batch_Norm_5_256/batchnorm/mul/ReadVariableOpReadVariableOpCtree_encoder_batch_norm_5_256_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02<
:Tree_Encoder/Batch_Norm_5_256/batchnorm/mul/ReadVariableOp■
+Tree_Encoder/Batch_Norm_5_256/batchnorm/mulMul1Tree_Encoder/Batch_Norm_5_256/batchnorm/Rsqrt:y:0BTree_Encoder/Batch_Norm_5_256/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2-
+Tree_Encoder/Batch_Norm_5_256/batchnorm/mulД
-Tree_Encoder/Batch_Norm_5_256/batchnorm/mul_1Mul,Tree_Encoder/Conv1D_5_256/IdentityN:output:0/Tree_Encoder/Batch_Norm_5_256/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2/
-Tree_Encoder/Batch_Norm_5_256/batchnorm/mul_1є
8Tree_Encoder/Batch_Norm_5_256/batchnorm/ReadVariableOp_1ReadVariableOpAtree_encoder_batch_norm_5_256_batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02:
8Tree_Encoder/Batch_Norm_5_256/batchnorm/ReadVariableOp_1■
-Tree_Encoder/Batch_Norm_5_256/batchnorm/mul_2Mul@Tree_Encoder/Batch_Norm_5_256/batchnorm/ReadVariableOp_1:value:0/Tree_Encoder/Batch_Norm_5_256/batchnorm/mul:z:0*
T0*
_output_shapes	
:А2/
-Tree_Encoder/Batch_Norm_5_256/batchnorm/mul_2є
8Tree_Encoder/Batch_Norm_5_256/batchnorm/ReadVariableOp_2ReadVariableOpAtree_encoder_batch_norm_5_256_batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02:
8Tree_Encoder/Batch_Norm_5_256/batchnorm/ReadVariableOp_2№
+Tree_Encoder/Batch_Norm_5_256/batchnorm/subSub@Tree_Encoder/Batch_Norm_5_256/batchnorm/ReadVariableOp_2:value:01Tree_Encoder/Batch_Norm_5_256/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2-
+Tree_Encoder/Batch_Norm_5_256/batchnorm/subЛ
-Tree_Encoder/Batch_Norm_5_256/batchnorm/add_1AddV21Tree_Encoder/Batch_Norm_5_256/batchnorm/mul_1:z:0/Tree_Encoder/Batch_Norm_5_256/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2/
-Tree_Encoder/Batch_Norm_5_256/batchnorm/add_1э
6Tree_Encoder/Batch_Norm_3_256/batchnorm/ReadVariableOpReadVariableOp?tree_encoder_batch_norm_3_256_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype028
6Tree_Encoder/Batch_Norm_3_256/batchnorm/ReadVariableOpг
-Tree_Encoder/Batch_Norm_3_256/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2/
-Tree_Encoder/Batch_Norm_3_256/batchnorm/add/yБ
+Tree_Encoder/Batch_Norm_3_256/batchnorm/addAddV2>Tree_Encoder/Batch_Norm_3_256/batchnorm/ReadVariableOp:value:06Tree_Encoder/Batch_Norm_3_256/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2-
+Tree_Encoder/Batch_Norm_3_256/batchnorm/add╛
-Tree_Encoder/Batch_Norm_3_256/batchnorm/RsqrtRsqrt/Tree_Encoder/Batch_Norm_3_256/batchnorm/add:z:0*
T0*
_output_shapes	
:А2/
-Tree_Encoder/Batch_Norm_3_256/batchnorm/Rsqrt∙
:Tree_Encoder/Batch_Norm_3_256/batchnorm/mul/ReadVariableOpReadVariableOpCtree_encoder_batch_norm_3_256_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02<
:Tree_Encoder/Batch_Norm_3_256/batchnorm/mul/ReadVariableOp■
+Tree_Encoder/Batch_Norm_3_256/batchnorm/mulMul1Tree_Encoder/Batch_Norm_3_256/batchnorm/Rsqrt:y:0BTree_Encoder/Batch_Norm_3_256/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2-
+Tree_Encoder/Batch_Norm_3_256/batchnorm/mulД
-Tree_Encoder/Batch_Norm_3_256/batchnorm/mul_1Mul,Tree_Encoder/Conv1D_3_256/IdentityN:output:0/Tree_Encoder/Batch_Norm_3_256/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2/
-Tree_Encoder/Batch_Norm_3_256/batchnorm/mul_1є
8Tree_Encoder/Batch_Norm_3_256/batchnorm/ReadVariableOp_1ReadVariableOpAtree_encoder_batch_norm_3_256_batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02:
8Tree_Encoder/Batch_Norm_3_256/batchnorm/ReadVariableOp_1■
-Tree_Encoder/Batch_Norm_3_256/batchnorm/mul_2Mul@Tree_Encoder/Batch_Norm_3_256/batchnorm/ReadVariableOp_1:value:0/Tree_Encoder/Batch_Norm_3_256/batchnorm/mul:z:0*
T0*
_output_shapes	
:А2/
-Tree_Encoder/Batch_Norm_3_256/batchnorm/mul_2є
8Tree_Encoder/Batch_Norm_3_256/batchnorm/ReadVariableOp_2ReadVariableOpAtree_encoder_batch_norm_3_256_batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02:
8Tree_Encoder/Batch_Norm_3_256/batchnorm/ReadVariableOp_2№
+Tree_Encoder/Batch_Norm_3_256/batchnorm/subSub@Tree_Encoder/Batch_Norm_3_256/batchnorm/ReadVariableOp_2:value:01Tree_Encoder/Batch_Norm_3_256/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2-
+Tree_Encoder/Batch_Norm_3_256/batchnorm/subЛ
-Tree_Encoder/Batch_Norm_3_256/batchnorm/add_1AddV21Tree_Encoder/Batch_Norm_3_256/batchnorm/mul_1:z:0/Tree_Encoder/Batch_Norm_3_256/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2/
-Tree_Encoder/Batch_Norm_3_256/batchnorm/add_1а
-Tree_Encoder/Max_Pool_3/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2/
-Tree_Encoder/Max_Pool_3/Max/reduction_indices▀
Tree_Encoder/Max_Pool_3/MaxMax1Tree_Encoder/Batch_Norm_3_256/batchnorm/add_1:z:06Tree_Encoder/Max_Pool_3/Max/reduction_indices:output:0*
T0*(
_output_shapes
:         А2
Tree_Encoder/Max_Pool_3/Maxа
-Tree_Encoder/Max_Pool_5/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2/
-Tree_Encoder/Max_Pool_5/Max/reduction_indices▀
Tree_Encoder/Max_Pool_5/MaxMax1Tree_Encoder/Batch_Norm_5_256/batchnorm/add_1:z:06Tree_Encoder/Max_Pool_5/Max/reduction_indices:output:0*
T0*(
_output_shapes
:         А2
Tree_Encoder/Max_Pool_5/Maxа
-Tree_Encoder/Max_Pool_7/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2/
-Tree_Encoder/Max_Pool_7/Max/reduction_indices▀
Tree_Encoder/Max_Pool_7/MaxMax1Tree_Encoder/Batch_Norm_7_256/batchnorm/add_1:z:06Tree_Encoder/Max_Pool_7/Max/reduction_indices:output:0*
T0*(
_output_shapes
:         А2
Tree_Encoder/Max_Pool_7/MaxШ
)Tree_Encoder/Pool_Concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2+
)Tree_Encoder/Pool_Concatenate/concat/axis║
$Tree_Encoder/Pool_Concatenate/concatConcatV2$Tree_Encoder/Max_Pool_3/Max:output:0$Tree_Encoder/Max_Pool_5/Max:output:0$Tree_Encoder/Max_Pool_7/Max:output:02Tree_Encoder/Pool_Concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:         А2&
$Tree_Encoder/Pool_Concatenate/concat─
)Tree_Encoder/Concatenate_Dropout/IdentityIdentity-Tree_Encoder/Pool_Concatenate/concat:output:0*
T0*(
_output_shapes
:         А2+
)Tree_Encoder/Concatenate_Dropout/Identity╓
-Tree_Encoder/Prediction/MatMul/ReadVariableOpReadVariableOp6tree_encoder_prediction_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02/
-Tree_Encoder/Prediction/MatMul/ReadVariableOpч
Tree_Encoder/Prediction/MatMulMatMul2Tree_Encoder/Concatenate_Dropout/Identity:output:05Tree_Encoder/Prediction/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2 
Tree_Encoder/Prediction/MatMul╘
.Tree_Encoder/Prediction/BiasAdd/ReadVariableOpReadVariableOp7tree_encoder_prediction_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.Tree_Encoder/Prediction/BiasAdd/ReadVariableOpс
Tree_Encoder/Prediction/BiasAddBiasAdd(Tree_Encoder/Prediction/MatMul:product:06Tree_Encoder/Prediction/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2!
Tree_Encoder/Prediction/BiasAddй
Tree_Encoder/Prediction/SigmoidSigmoid(Tree_Encoder/Prediction/BiasAdd:output:0*
T0*'
_output_shapes
:         2!
Tree_Encoder/Prediction/SigmoidМ
IdentityIdentity#Tree_Encoder/Prediction/Sigmoid:y:07^Tree_Encoder/Batch_Norm_3_128/batchnorm/ReadVariableOp9^Tree_Encoder/Batch_Norm_3_128/batchnorm/ReadVariableOp_19^Tree_Encoder/Batch_Norm_3_128/batchnorm/ReadVariableOp_2;^Tree_Encoder/Batch_Norm_3_128/batchnorm/mul/ReadVariableOp7^Tree_Encoder/Batch_Norm_3_256/batchnorm/ReadVariableOp9^Tree_Encoder/Batch_Norm_3_256/batchnorm/ReadVariableOp_19^Tree_Encoder/Batch_Norm_3_256/batchnorm/ReadVariableOp_2;^Tree_Encoder/Batch_Norm_3_256/batchnorm/mul/ReadVariableOp7^Tree_Encoder/Batch_Norm_5_128/batchnorm/ReadVariableOp9^Tree_Encoder/Batch_Norm_5_128/batchnorm/ReadVariableOp_19^Tree_Encoder/Batch_Norm_5_128/batchnorm/ReadVariableOp_2;^Tree_Encoder/Batch_Norm_5_128/batchnorm/mul/ReadVariableOp7^Tree_Encoder/Batch_Norm_5_256/batchnorm/ReadVariableOp9^Tree_Encoder/Batch_Norm_5_256/batchnorm/ReadVariableOp_19^Tree_Encoder/Batch_Norm_5_256/batchnorm/ReadVariableOp_2;^Tree_Encoder/Batch_Norm_5_256/batchnorm/mul/ReadVariableOp7^Tree_Encoder/Batch_Norm_7_128/batchnorm/ReadVariableOp9^Tree_Encoder/Batch_Norm_7_128/batchnorm/ReadVariableOp_19^Tree_Encoder/Batch_Norm_7_128/batchnorm/ReadVariableOp_2;^Tree_Encoder/Batch_Norm_7_128/batchnorm/mul/ReadVariableOp7^Tree_Encoder/Batch_Norm_7_256/batchnorm/ReadVariableOp9^Tree_Encoder/Batch_Norm_7_256/batchnorm/ReadVariableOp_19^Tree_Encoder/Batch_Norm_7_256/batchnorm/ReadVariableOp_2;^Tree_Encoder/Batch_Norm_7_256/batchnorm/mul/ReadVariableOp1^Tree_Encoder/Conv1D_3_128/BiasAdd/ReadVariableOp=^Tree_Encoder/Conv1D_3_128/conv1d/ExpandDims_1/ReadVariableOp1^Tree_Encoder/Conv1D_3_256/BiasAdd/ReadVariableOp=^Tree_Encoder/Conv1D_3_256/conv1d/ExpandDims_1/ReadVariableOp1^Tree_Encoder/Conv1D_5_128/BiasAdd/ReadVariableOp=^Tree_Encoder/Conv1D_5_128/conv1d/ExpandDims_1/ReadVariableOp1^Tree_Encoder/Conv1D_5_256/BiasAdd/ReadVariableOp=^Tree_Encoder/Conv1D_5_256/conv1d/ExpandDims_1/ReadVariableOp1^Tree_Encoder/Conv1D_7_128/BiasAdd/ReadVariableOp=^Tree_Encoder/Conv1D_7_128/conv1d/ExpandDims_1/ReadVariableOp1^Tree_Encoder/Conv1D_7_256/BiasAdd/ReadVariableOp=^Tree_Encoder/Conv1D_7_256/conv1d/ExpandDims_1/ReadVariableOp/^Tree_Encoder/Prediction/BiasAdd/ReadVariableOp.^Tree_Encoder/Prediction/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*А
_input_shapeso
m:                  А: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2p
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
!:                  А
 
_user_specified_nameInputs
№
а
L__inference_Conv1D_7_128_layer_call_and_return_conditional_losses_1294350755

inputsC
+conv1d_expanddims_1_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А

identity_1ИвBiasAdd/ReadVariableOpв"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimа
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#                  А2
conv1d/ExpandDims║
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:АА*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim╣
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:АА2
conv1d/ExpandDims_1└
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#                  А*
paddingSAME*
strides
2
conv1dЬ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*5
_output_shapes#
!:                  А*
squeeze_dims

¤        2
conv1d/SqueezeН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЦ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:                  А2	
BiasAddo
SigmoidSigmoidBiasAdd:output:0*
T0*5
_output_shapes#
!:                  А2	
Sigmoidp
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*5
_output_shapes#
!:                  А2
muli
IdentityIdentitymul:z:0*
T0*5
_output_shapes#
!:                  А2

Identity╒
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1294350748*V
_output_shapesD
B:                  А:                  А2
	IdentityN╢

Identity_1IdentityIdentityN:output:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*5
_output_shapes#
!:                  А2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:                  А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
╛
╘
5__inference_Batch_Norm_5_128_layer_call_fn_1294350848

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_Batch_Norm_5_128_layer_call_and_return_conditional_losses_12943480212
StatefulPartitionedCallЬ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
н
Э
/__inference_Prediction_layer_call_fn_1294351376

inputs
unknown:	А
	unknown_0:
identityИвStatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_Prediction_layer_call_and_return_conditional_losses_12943491072
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
й
f
J__inference_Max_Pool_3_layer_call_and_return_conditional_losses_1294348814

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
:                  2
Maxi
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:                  2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
№
а
L__inference_Conv1D_5_128_layer_call_and_return_conditional_losses_1294350725

inputsC
+conv1d_expanddims_1_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А

identity_1ИвBiasAdd/ReadVariableOpв"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimа
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#                  А2
conv1d/ExpandDims║
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:АА*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim╣
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:АА2
conv1d/ExpandDims_1└
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#                  А*
paddingSAME*
strides
2
conv1dЬ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*5
_output_shapes#
!:                  А*
squeeze_dims

¤        2
conv1d/SqueezeН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЦ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:                  А2	
BiasAddo
SigmoidSigmoidBiasAdd:output:0*
T0*5
_output_shapes#
!:                  А2	
Sigmoidp
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*5
_output_shapes#
!:                  А2
muli
IdentityIdentitymul:z:0*
T0*5
_output_shapes#
!:                  А2

Identity╒
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1294350718*V
_output_shapesD
B:                  А:                  А2
	IdentityN╢

Identity_1IdentityIdentityN:output:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*5
_output_shapes#
!:                  А2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:                  А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
╛
╘
5__inference_Batch_Norm_5_256_layer_call_fn_1294351178

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_Batch_Norm_5_256_layer_call_and_return_conditional_losses_12943485072
StatefulPartitionedCallЬ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
Д
q
S__inference_Concatenate_Dropout_layer_call_and_return_conditional_losses_1294349094

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         А2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
№
а
L__inference_Conv1D_3_128_layer_call_and_return_conditional_losses_1294348935

inputsC
+conv1d_expanddims_1_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А

identity_1ИвBiasAdd/ReadVariableOpв"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimа
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#                  А2
conv1d/ExpandDims║
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:АА*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim╣
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:АА2
conv1d/ExpandDims_1└
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#                  А*
paddingSAME*
strides
2
conv1dЬ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*5
_output_shapes#
!:                  А*
squeeze_dims

¤        2
conv1d/SqueezeН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЦ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:                  А2	
BiasAddo
SigmoidSigmoidBiasAdd:output:0*
T0*5
_output_shapes#
!:                  А2	
Sigmoidp
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*5
_output_shapes#
!:                  А2
muli
IdentityIdentitymul:z:0*
T0*5
_output_shapes#
!:                  А2

Identity╒
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*0
_gradient_op_typeCustomGradient-1294348928*V
_output_shapesD
B:                  А:                  А2
	IdentityN╢

Identity_1IdentityIdentityN:output:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*5
_output_shapes#
!:                  А2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:                  А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
у
o
5__inference_Pool_Concatenate_layer_call_fn_1294351332
inputs_0
inputs_1
inputs_2
identityъ
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_Pool_Concatenate_layer_call_and_return_conditional_losses_12943490872
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:         А:         А:         А:R N
(
_output_shapes
:         А
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:         А
"
_user_specified_name
inputs/1:RN
(
_output_shapes
:         А
"
_user_specified_name
inputs/2
№*
э
P__inference_Batch_Norm_5_128_layer_call_and_return_conditional_losses_1294350915

inputs6
'assignmovingavg_readvariableop_resource:	А8
)assignmovingavg_1_readvariableop_resource:	А4
%batchnorm_mul_readvariableop_resource:	А0
!batchnorm_readvariableop_resource:	А
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesФ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/meanБ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:А2
moments/StopGradient▓
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:                  А2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╖
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:А*
	keep_dims(2
moments/varianceВ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/SqueezeК
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayе
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOpЩ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:А2
AssignMovingAvg/subР
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:А2
AssignMovingAvg/mul┐
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
╫#<2
AssignMovingAvg_1/decayл
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOpб
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:А2
AssignMovingAvg_1/subШ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:А2
AssignMovingAvg_1/mul╔
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
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/add_1Щ
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  А: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
ё
│
P__inference_Batch_Norm_5_256_layer_call_and_return_conditional_losses_1294351211

inputs0
!batchnorm_readvariableop_resource:	А4
%batchnorm_mul_readvariableop_resource:	А2
#batchnorm_readvariableop_1_resource:	А2
#batchnorm_readvariableop_2_resource:	А
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulД
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subУ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:                  А2
batchnorm/add_1щ
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:                  А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):                  А: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:                  А
 
_user_specified_nameinputs
ЖR
╕
#__inference__traced_save_1294351524
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
*savev2_prediction_bias_read_readvariableop
savev2_const

identity_1ИвMergeV2CheckpointsП
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
Const_1Л
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
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameп
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:'*
dtype0*┴
value╖B┤'B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names╓
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:'*
dtype0*a
valueXBV'B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesР
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0.savev2_conv1d_3_128_kernel_read_readvariableop,savev2_conv1d_3_128_bias_read_readvariableop.savev2_conv1d_5_128_kernel_read_readvariableop,savev2_conv1d_5_128_bias_read_readvariableop.savev2_conv1d_7_128_kernel_read_readvariableop,savev2_conv1d_7_128_bias_read_readvariableop1savev2_batch_norm_3_128_gamma_read_readvariableop0savev2_batch_norm_3_128_beta_read_readvariableop7savev2_batch_norm_3_128_moving_mean_read_readvariableop;savev2_batch_norm_3_128_moving_variance_read_readvariableop1savev2_batch_norm_5_128_gamma_read_readvariableop0savev2_batch_norm_5_128_beta_read_readvariableop7savev2_batch_norm_5_128_moving_mean_read_readvariableop;savev2_batch_norm_5_128_moving_variance_read_readvariableop1savev2_batch_norm_7_128_gamma_read_readvariableop0savev2_batch_norm_7_128_beta_read_readvariableop7savev2_batch_norm_7_128_moving_mean_read_readvariableop;savev2_batch_norm_7_128_moving_variance_read_readvariableop.savev2_conv1d_3_256_kernel_read_readvariableop,savev2_conv1d_3_256_bias_read_readvariableop.savev2_conv1d_5_256_kernel_read_readvariableop,savev2_conv1d_5_256_bias_read_readvariableop.savev2_conv1d_7_256_kernel_read_readvariableop,savev2_conv1d_7_256_bias_read_readvariableop1savev2_batch_norm_3_256_gamma_read_readvariableop0savev2_batch_norm_3_256_beta_read_readvariableop7savev2_batch_norm_3_256_moving_mean_read_readvariableop;savev2_batch_norm_3_256_moving_variance_read_readvariableop1savev2_batch_norm_5_256_gamma_read_readvariableop0savev2_batch_norm_5_256_beta_read_readvariableop7savev2_batch_norm_5_256_moving_mean_read_readvariableop;savev2_batch_norm_5_256_moving_variance_read_readvariableop1savev2_batch_norm_7_256_gamma_read_readvariableop0savev2_batch_norm_7_256_beta_read_readvariableop7savev2_batch_norm_7_256_moving_mean_read_readvariableop;savev2_batch_norm_7_256_moving_variance_read_readvariableop,savev2_prediction_kernel_read_readvariableop*savev2_prediction_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *5
dtypes+
)2'2
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesб
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

identity_1Identity_1:output:0*▄
_input_shapes╩
╟: :АА:А:АА:А:АА:А:А:А:А:А:А:А:А:А:А:А:А:А:АА:А:АА:А:АА:А:А:А:А:А:А:А:А:А:А:А:А:А:	А:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:*&
$
_output_shapes
:АА:!

_output_shapes	
:А:*&
$
_output_shapes
:АА:!

_output_shapes	
:А:*&
$
_output_shapes
:АА:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!	

_output_shapes	
:А:!


_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:*&
$
_output_shapes
:АА:!

_output_shapes	
:А:*&
$
_output_shapes
:АА:!

_output_shapes	
:А:*&
$
_output_shapes
:АА:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:! 

_output_shapes	
:А:!!

_output_shapes	
:А:!"

_output_shapes	
:А:!#

_output_shapes	
:А:!$

_output_shapes	
:А:%%!

_output_shapes
:	А: &

_output_shapes
::'

_output_shapes
: 
Й
K
/__inference_Max_Pool_3_layer_call_fn_1294348820

inputs
identity╘
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:                  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_Max_Pool_3_layer_call_and_return_conditional_losses_12943488142
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:                  2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs"╠L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*╣
serving_defaultе
G
Inputs=
serving_default_Inputs:0                  А>

Prediction0
StatefulPartitionedCall:0         tensorflow/serving/predict:ЮЮ
└╗
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
trainable_variables
regularization_losses
	variables
	keras_api

signatures
Ў__call__
+ў&call_and_return_all_conditional_losses
°_default_save_signature"Э╡
_tf_keras_networkА╡{"name": "Tree_Encoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "Tree_Encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Inputs"}, "name": "Inputs", "inbound_nodes": []}, {"class_name": "Dropout", "config": {"name": "Embedding_Dropout", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "Embedding_Dropout", "inbound_nodes": [[["Inputs", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "Conv1D_3_128", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_3_128", "inbound_nodes": [[["Embedding_Dropout", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "Conv1D_5_128", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_5_128", "inbound_nodes": [[["Embedding_Dropout", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "Conv1D_7_128", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [7]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_7_128", "inbound_nodes": [[["Embedding_Dropout", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "Batch_Norm_3_128", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "Batch_Norm_3_128", "inbound_nodes": [[["Conv1D_3_128", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "Batch_Norm_5_128", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "Batch_Norm_5_128", "inbound_nodes": [[["Conv1D_5_128", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "Batch_Norm_7_128", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "Batch_Norm_7_128", "inbound_nodes": [[["Conv1D_7_128", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "Conv1D_3_256", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_3_256", "inbound_nodes": [[["Batch_Norm_3_128", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "Conv1D_5_256", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_5_256", "inbound_nodes": [[["Batch_Norm_5_128", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "Conv1D_7_256", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [7]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_7_256", "inbound_nodes": [[["Batch_Norm_7_128", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "Batch_Norm_3_256", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "Batch_Norm_3_256", "inbound_nodes": [[["Conv1D_3_256", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "Batch_Norm_5_256", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "Batch_Norm_5_256", "inbound_nodes": [[["Conv1D_5_256", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "Batch_Norm_7_256", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "Batch_Norm_7_256", "inbound_nodes": [[["Conv1D_7_256", 0, 0, {}]]]}, {"class_name": "GlobalMaxPooling1D", "config": {"name": "Max_Pool_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "Max_Pool_3", "inbound_nodes": [[["Batch_Norm_3_256", 0, 0, {}]]]}, {"class_name": "GlobalMaxPooling1D", "config": {"name": "Max_Pool_5", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "Max_Pool_5", "inbound_nodes": [[["Batch_Norm_5_256", 0, 0, {}]]]}, {"class_name": "GlobalMaxPooling1D", "config": {"name": "Max_Pool_7", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "Max_Pool_7", "inbound_nodes": [[["Batch_Norm_7_256", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "Pool_Concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "Pool_Concatenate", "inbound_nodes": [[["Max_Pool_3", 0, 0, {}], ["Max_Pool_5", 0, 0, {}], ["Max_Pool_7", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "Concatenate_Dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "Concatenate_Dropout", "inbound_nodes": [[["Pool_Concatenate", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Prediction", "trainable": true, "dtype": "float32", "units": 28, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Prediction", "inbound_nodes": [[["Concatenate_Dropout", 0, 0, {}]]]}], "input_layers": [["Inputs", 0, 0]], "output_layers": [["Prediction", 0, 0]]}, "shared_object_id": 58, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 128]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 128]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, null, 128]}, "float32", "Inputs"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "Tree_Encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Inputs"}, "name": "Inputs", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Dropout", "config": {"name": "Embedding_Dropout", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "Embedding_Dropout", "inbound_nodes": [[["Inputs", 0, 0, {}]]], "shared_object_id": 1}, {"class_name": "Conv1D", "config": {"name": "Conv1D_3_128", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 2}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 3}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_3_128", "inbound_nodes": [[["Embedding_Dropout", 0, 0, {}]]], "shared_object_id": 4}, {"class_name": "Conv1D", "config": {"name": "Conv1D_5_128", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_5_128", "inbound_nodes": [[["Embedding_Dropout", 0, 0, {}]]], "shared_object_id": 7}, {"class_name": "Conv1D", "config": {"name": "Conv1D_7_128", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [7]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_7_128", "inbound_nodes": [[["Embedding_Dropout", 0, 0, {}]]], "shared_object_id": 10}, {"class_name": "BatchNormalization", "config": {"name": "Batch_Norm_3_128", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 12}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 14}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "Batch_Norm_3_128", "inbound_nodes": [[["Conv1D_3_128", 0, 0, {}]]], "shared_object_id": 15}, {"class_name": "BatchNormalization", "config": {"name": "Batch_Norm_5_128", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 17}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 18}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 19}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "Batch_Norm_5_128", "inbound_nodes": [[["Conv1D_5_128", 0, 0, {}]]], "shared_object_id": 20}, {"class_name": "BatchNormalization", "config": {"name": "Batch_Norm_7_128", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 22}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 24}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "Batch_Norm_7_128", "inbound_nodes": [[["Conv1D_7_128", 0, 0, {}]]], "shared_object_id": 25}, {"class_name": "Conv1D", "config": {"name": "Conv1D_3_256", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 26}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 27}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_3_256", "inbound_nodes": [[["Batch_Norm_3_128", 0, 0, {}]]], "shared_object_id": 28}, {"class_name": "Conv1D", "config": {"name": "Conv1D_5_256", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 29}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 30}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_5_256", "inbound_nodes": [[["Batch_Norm_5_128", 0, 0, {}]]], "shared_object_id": 31}, {"class_name": "Conv1D", "config": {"name": "Conv1D_7_256", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [7]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 32}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 33}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_7_256", "inbound_nodes": [[["Batch_Norm_7_128", 0, 0, {}]]], "shared_object_id": 34}, {"class_name": "BatchNormalization", "config": {"name": "Batch_Norm_3_256", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 35}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 36}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 37}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 38}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "Batch_Norm_3_256", "inbound_nodes": [[["Conv1D_3_256", 0, 0, {}]]], "shared_object_id": 39}, {"class_name": "BatchNormalization", "config": {"name": "Batch_Norm_5_256", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 40}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 41}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 42}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 43}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "Batch_Norm_5_256", "inbound_nodes": [[["Conv1D_5_256", 0, 0, {}]]], "shared_object_id": 44}, {"class_name": "BatchNormalization", "config": {"name": "Batch_Norm_7_256", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 45}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 46}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 47}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 48}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "Batch_Norm_7_256", "inbound_nodes": [[["Conv1D_7_256", 0, 0, {}]]], "shared_object_id": 49}, {"class_name": "GlobalMaxPooling1D", "config": {"name": "Max_Pool_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "Max_Pool_3", "inbound_nodes": [[["Batch_Norm_3_256", 0, 0, {}]]], "shared_object_id": 50}, {"class_name": "GlobalMaxPooling1D", "config": {"name": "Max_Pool_5", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "Max_Pool_5", "inbound_nodes": [[["Batch_Norm_5_256", 0, 0, {}]]], "shared_object_id": 51}, {"class_name": "GlobalMaxPooling1D", "config": {"name": "Max_Pool_7", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "Max_Pool_7", "inbound_nodes": [[["Batch_Norm_7_256", 0, 0, {}]]], "shared_object_id": 52}, {"class_name": "Concatenate", "config": {"name": "Pool_Concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "Pool_Concatenate", "inbound_nodes": [[["Max_Pool_3", 0, 0, {}], ["Max_Pool_5", 0, 0, {}], ["Max_Pool_7", 0, 0, {}]]], "shared_object_id": 53}, {"class_name": "Dropout", "config": {"name": "Concatenate_Dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "Concatenate_Dropout", "inbound_nodes": [[["Pool_Concatenate", 0, 0, {}]]], "shared_object_id": 54}, {"class_name": "Dense", "config": {"name": "Prediction", "trainable": true, "dtype": "float32", "units": 28, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 55}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 56}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Prediction", "inbound_nodes": [[["Concatenate_Dropout", 0, 0, {}]]], "shared_object_id": 57}], "input_layers": [["Inputs", 0, 0]], "output_layers": [["Prediction", 0, 0]]}}}
ў"Ї
_tf_keras_input_layer╘{"class_name": "InputLayer", "name": "Inputs", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 128]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Inputs"}}
╣
trainable_variables
regularization_losses
	variables
	keras_api
∙__call__
+·&call_and_return_all_conditional_losses"и
_tf_keras_layerО{"name": "Embedding_Dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "Embedding_Dropout", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "inbound_nodes": [[["Inputs", 0, 0, {}]]], "shared_object_id": 1}
И

kernel
bias
 trainable_variables
!regularization_losses
"	variables
#	keras_api
√__call__
+№&call_and_return_all_conditional_losses"с	
_tf_keras_layer╟	{"name": "Conv1D_3_128", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "Conv1D_3_128", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 2}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 3}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["Embedding_Dropout", 0, 0, {}]]], "shared_object_id": 4, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 128}}, "shared_object_id": 60}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 128]}}
И

$kernel
%bias
&trainable_variables
'regularization_losses
(	variables
)	keras_api
¤__call__
+■&call_and_return_all_conditional_losses"с	
_tf_keras_layer╟	{"name": "Conv1D_5_128", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "Conv1D_5_128", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["Embedding_Dropout", 0, 0, {}]]], "shared_object_id": 7, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 128}}, "shared_object_id": 61}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 128]}}
Й

*kernel
+bias
,trainable_variables
-regularization_losses
.	variables
/	keras_api
 __call__
+А&call_and_return_all_conditional_losses"т	
_tf_keras_layer╚	{"name": "Conv1D_7_128", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "Conv1D_7_128", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [7]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["Embedding_Dropout", 0, 0, {}]]], "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 128}}, "shared_object_id": 62}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 128]}}
є

0axis
	1gamma
2beta
3moving_mean
4moving_variance
5trainable_variables
6regularization_losses
7	variables
8	keras_api
Б__call__
+В&call_and_return_all_conditional_losses"Э	
_tf_keras_layerГ	{"name": "Batch_Norm_3_128", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "Batch_Norm_3_128", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 12}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 14}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["Conv1D_3_128", 0, 0, {}]]], "shared_object_id": 15, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 128}}, "shared_object_id": 63}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 128]}}
є

9axis
	:gamma
;beta
<moving_mean
=moving_variance
>trainable_variables
?regularization_losses
@	variables
A	keras_api
Г__call__
+Д&call_and_return_all_conditional_losses"Э	
_tf_keras_layerГ	{"name": "Batch_Norm_5_128", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "Batch_Norm_5_128", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 17}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 18}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 19}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["Conv1D_5_128", 0, 0, {}]]], "shared_object_id": 20, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 128}}, "shared_object_id": 64}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 128]}}
є

Baxis
	Cgamma
Dbeta
Emoving_mean
Fmoving_variance
Gtrainable_variables
Hregularization_losses
I	variables
J	keras_api
Е__call__
+Ж&call_and_return_all_conditional_losses"Э	
_tf_keras_layerГ	{"name": "Batch_Norm_7_128", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "Batch_Norm_7_128", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 22}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 24}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["Conv1D_7_128", 0, 0, {}]]], "shared_object_id": 25, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 128}}, "shared_object_id": 65}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 128]}}
К

Kkernel
Lbias
Mtrainable_variables
Nregularization_losses
O	variables
P	keras_api
З__call__
+И&call_and_return_all_conditional_losses"у	
_tf_keras_layer╔	{"name": "Conv1D_3_256", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "Conv1D_3_256", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 26}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 27}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["Batch_Norm_3_128", 0, 0, {}]]], "shared_object_id": 28, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 128}}, "shared_object_id": 66}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 128]}}
К

Qkernel
Rbias
Strainable_variables
Tregularization_losses
U	variables
V	keras_api
Й__call__
+К&call_and_return_all_conditional_losses"у	
_tf_keras_layer╔	{"name": "Conv1D_5_256", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "Conv1D_5_256", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 29}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 30}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["Batch_Norm_5_128", 0, 0, {}]]], "shared_object_id": 31, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 128}}, "shared_object_id": 67}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 128]}}
К

Wkernel
Xbias
Ytrainable_variables
Zregularization_losses
[	variables
\	keras_api
Л__call__
+М&call_and_return_all_conditional_losses"у	
_tf_keras_layer╔	{"name": "Conv1D_7_256", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "Conv1D_7_256", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [7]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 32}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 33}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["Batch_Norm_7_128", 0, 0, {}]]], "shared_object_id": 34, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 128}}, "shared_object_id": 68}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 128]}}
є

]axis
	^gamma
_beta
`moving_mean
amoving_variance
btrainable_variables
cregularization_losses
d	variables
e	keras_api
Н__call__
+О&call_and_return_all_conditional_losses"Э	
_tf_keras_layerГ	{"name": "Batch_Norm_3_256", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "Batch_Norm_3_256", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 35}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 36}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 37}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 38}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["Conv1D_3_256", 0, 0, {}]]], "shared_object_id": 39, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 256}}, "shared_object_id": 69}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 256]}}
є

faxis
	ggamma
hbeta
imoving_mean
jmoving_variance
ktrainable_variables
lregularization_losses
m	variables
n	keras_api
П__call__
+Р&call_and_return_all_conditional_losses"Э	
_tf_keras_layerГ	{"name": "Batch_Norm_5_256", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "Batch_Norm_5_256", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 40}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 41}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 42}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 43}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["Conv1D_5_256", 0, 0, {}]]], "shared_object_id": 44, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 256}}, "shared_object_id": 70}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 256]}}
є

oaxis
	pgamma
qbeta
rmoving_mean
smoving_variance
ttrainable_variables
uregularization_losses
v	variables
w	keras_api
С__call__
+Т&call_and_return_all_conditional_losses"Э	
_tf_keras_layerГ	{"name": "Batch_Norm_7_256", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "Batch_Norm_7_256", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 45}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 46}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 47}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 48}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["Conv1D_7_256", 0, 0, {}]]], "shared_object_id": 49, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 256}}, "shared_object_id": 71}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 256]}}
┌
xtrainable_variables
yregularization_losses
z	variables
{	keras_api
У__call__
+Ф&call_and_return_all_conditional_losses"╔
_tf_keras_layerп{"name": "Max_Pool_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "GlobalMaxPooling1D", "config": {"name": "Max_Pool_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["Batch_Norm_3_256", 0, 0, {}]]], "shared_object_id": 50, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 72}}
┌
|trainable_variables
}regularization_losses
~	variables
	keras_api
Х__call__
+Ц&call_and_return_all_conditional_losses"╔
_tf_keras_layerп{"name": "Max_Pool_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "GlobalMaxPooling1D", "config": {"name": "Max_Pool_5", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["Batch_Norm_5_256", 0, 0, {}]]], "shared_object_id": 51, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 73}}
▐
Аtrainable_variables
Бregularization_losses
В	variables
Г	keras_api
Ч__call__
+Ш&call_and_return_all_conditional_losses"╔
_tf_keras_layerп{"name": "Max_Pool_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "GlobalMaxPooling1D", "config": {"name": "Max_Pool_7", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["Batch_Norm_7_256", 0, 0, {}]]], "shared_object_id": 52, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 74}}
Л
Дtrainable_variables
Еregularization_losses
Ж	variables
З	keras_api
Щ__call__
+Ъ&call_and_return_all_conditional_losses"Ў
_tf_keras_layer▄{"name": "Pool_Concatenate", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Concatenate", "config": {"name": "Pool_Concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "inbound_nodes": [[["Max_Pool_3", 0, 0, {}], ["Max_Pool_5", 0, 0, {}], ["Max_Pool_7", 0, 0, {}]]], "shared_object_id": 53, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 256]}, {"class_name": "TensorShape", "items": [null, 256]}, {"class_name": "TensorShape", "items": [null, 256]}]}
╠
Иtrainable_variables
Йregularization_losses
К	variables
Л	keras_api
Ы__call__
+Ь&call_and_return_all_conditional_losses"╖
_tf_keras_layerЭ{"name": "Concatenate_Dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "Concatenate_Dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "inbound_nodes": [[["Pool_Concatenate", 0, 0, {}]]], "shared_object_id": 54}
Ы	
Мkernel
	Нbias
Оtrainable_variables
Пregularization_losses
Р	variables
С	keras_api
Э__call__
+Ю&call_and_return_all_conditional_losses"ю
_tf_keras_layer╘{"name": "Prediction", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "Prediction", "trainable": true, "dtype": "float32", "units": 28, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 55}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 56}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["Concatenate_Dropout", 0, 0, {}]]], "shared_object_id": 57, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 768}}, "shared_object_id": 75}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 768]}}
ш
0
1
$2
%3
*4
+5
16
27
:8
;9
C10
D11
K12
L13
Q14
R15
W16
X17
^18
_19
g20
h21
p22
q23
М24
Н25"
trackable_list_wrapper
 "
trackable_list_wrapper
╚
0
1
$2
%3
*4
+5
16
27
38
49
:10
;11
<12
=13
C14
D15
E16
F17
K18
L19
Q20
R21
W22
X23
^24
_25
`26
a27
g28
h29
i30
j31
p32
q33
r34
s35
М36
Н37"
trackable_list_wrapper
╙
trainable_variables
Тnon_trainable_variables
 Уlayer_regularization_losses
regularization_losses
	variables
Фmetrics
Хlayers
Цlayer_metrics
Ў__call__
°_default_save_signature
+ў&call_and_return_all_conditional_losses
'ў"call_and_return_conditional_losses"
_generic_user_object
-
Яserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
trainable_variables
Чnon_trainable_variables
 Шlayer_regularization_losses
regularization_losses
	variables
Щmetrics
Ъlayers
Ыlayer_metrics
∙__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses"
_generic_user_object
+:)АА2Conv1D_3_128/kernel
 :А2Conv1D_3_128/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
╡
 trainable_variables
Ьnon_trainable_variables
 Эlayer_regularization_losses
!regularization_losses
"	variables
Юmetrics
Яlayers
аlayer_metrics
√__call__
+№&call_and_return_all_conditional_losses
'№"call_and_return_conditional_losses"
_generic_user_object
+:)АА2Conv1D_5_128/kernel
 :А2Conv1D_5_128/bias
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
╡
&trainable_variables
бnon_trainable_variables
 вlayer_regularization_losses
'regularization_losses
(	variables
гmetrics
дlayers
еlayer_metrics
¤__call__
+■&call_and_return_all_conditional_losses
'■"call_and_return_conditional_losses"
_generic_user_object
+:)АА2Conv1D_7_128/kernel
 :А2Conv1D_7_128/bias
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
╡
,trainable_variables
жnon_trainable_variables
 зlayer_regularization_losses
-regularization_losses
.	variables
иmetrics
йlayers
кlayer_metrics
 __call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
%:#А2Batch_Norm_3_128/gamma
$:"А2Batch_Norm_3_128/beta
-:+А (2Batch_Norm_3_128/moving_mean
1:/А (2 Batch_Norm_3_128/moving_variance
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
<
10
21
32
43"
trackable_list_wrapper
╡
5trainable_variables
лnon_trainable_variables
 мlayer_regularization_losses
6regularization_losses
7	variables
нmetrics
оlayers
пlayer_metrics
Б__call__
+В&call_and_return_all_conditional_losses
'В"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
%:#А2Batch_Norm_5_128/gamma
$:"А2Batch_Norm_5_128/beta
-:+А (2Batch_Norm_5_128/moving_mean
1:/А (2 Batch_Norm_5_128/moving_variance
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
:0
;1
<2
=3"
trackable_list_wrapper
╡
>trainable_variables
░non_trainable_variables
 ▒layer_regularization_losses
?regularization_losses
@	variables
▓metrics
│layers
┤layer_metrics
Г__call__
+Д&call_and_return_all_conditional_losses
'Д"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
%:#А2Batch_Norm_7_128/gamma
$:"А2Batch_Norm_7_128/beta
-:+А (2Batch_Norm_7_128/moving_mean
1:/А (2 Batch_Norm_7_128/moving_variance
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
╡
Gtrainable_variables
╡non_trainable_variables
 ╢layer_regularization_losses
Hregularization_losses
I	variables
╖metrics
╕layers
╣layer_metrics
Е__call__
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses"
_generic_user_object
+:)АА2Conv1D_3_256/kernel
 :А2Conv1D_3_256/bias
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
╡
Mtrainable_variables
║non_trainable_variables
 ╗layer_regularization_losses
Nregularization_losses
O	variables
╝metrics
╜layers
╛layer_metrics
З__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses"
_generic_user_object
+:)АА2Conv1D_5_256/kernel
 :А2Conv1D_5_256/bias
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
╡
Strainable_variables
┐non_trainable_variables
 └layer_regularization_losses
Tregularization_losses
U	variables
┴metrics
┬layers
├layer_metrics
Й__call__
+К&call_and_return_all_conditional_losses
'К"call_and_return_conditional_losses"
_generic_user_object
+:)АА2Conv1D_7_256/kernel
 :А2Conv1D_7_256/bias
.
W0
X1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
╡
Ytrainable_variables
─non_trainable_variables
 ┼layer_regularization_losses
Zregularization_losses
[	variables
╞metrics
╟layers
╚layer_metrics
Л__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
%:#А2Batch_Norm_3_256/gamma
$:"А2Batch_Norm_3_256/beta
-:+А (2Batch_Norm_3_256/moving_mean
1:/А (2 Batch_Norm_3_256/moving_variance
.
^0
_1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
^0
_1
`2
a3"
trackable_list_wrapper
╡
btrainable_variables
╔non_trainable_variables
 ╩layer_regularization_losses
cregularization_losses
d	variables
╦metrics
╠layers
═layer_metrics
Н__call__
+О&call_and_return_all_conditional_losses
'О"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
%:#А2Batch_Norm_5_256/gamma
$:"А2Batch_Norm_5_256/beta
-:+А (2Batch_Norm_5_256/moving_mean
1:/А (2 Batch_Norm_5_256/moving_variance
.
g0
h1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
g0
h1
i2
j3"
trackable_list_wrapper
╡
ktrainable_variables
╬non_trainable_variables
 ╧layer_regularization_losses
lregularization_losses
m	variables
╨metrics
╤layers
╥layer_metrics
П__call__
+Р&call_and_return_all_conditional_losses
'Р"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
%:#А2Batch_Norm_7_256/gamma
$:"А2Batch_Norm_7_256/beta
-:+А (2Batch_Norm_7_256/moving_mean
1:/А (2 Batch_Norm_7_256/moving_variance
.
p0
q1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
p0
q1
r2
s3"
trackable_list_wrapper
╡
ttrainable_variables
╙non_trainable_variables
 ╘layer_regularization_losses
uregularization_losses
v	variables
╒metrics
╓layers
╫layer_metrics
С__call__
+Т&call_and_return_all_conditional_losses
'Т"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
xtrainable_variables
╪non_trainable_variables
 ┘layer_regularization_losses
yregularization_losses
z	variables
┌metrics
█layers
▄layer_metrics
У__call__
+Ф&call_and_return_all_conditional_losses
'Ф"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
|trainable_variables
▌non_trainable_variables
 ▐layer_regularization_losses
}regularization_losses
~	variables
▀metrics
рlayers
сlayer_metrics
Х__call__
+Ц&call_and_return_all_conditional_losses
'Ц"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Аtrainable_variables
тnon_trainable_variables
 уlayer_regularization_losses
Бregularization_losses
В	variables
фmetrics
хlayers
цlayer_metrics
Ч__call__
+Ш&call_and_return_all_conditional_losses
'Ш"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Дtrainable_variables
чnon_trainable_variables
 шlayer_regularization_losses
Еregularization_losses
Ж	variables
щmetrics
ъlayers
ыlayer_metrics
Щ__call__
+Ъ&call_and_return_all_conditional_losses
'Ъ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Иtrainable_variables
ьnon_trainable_variables
 эlayer_regularization_losses
Йregularization_losses
К	variables
юmetrics
яlayers
Ёlayer_metrics
Ы__call__
+Ь&call_and_return_all_conditional_losses
'Ь"call_and_return_conditional_losses"
_generic_user_object
$:"	А2Prediction/kernel
:2Prediction/bias
0
М0
Н1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
М0
Н1"
trackable_list_wrapper
╕
Оtrainable_variables
ёnon_trainable_variables
 Єlayer_regularization_losses
Пregularization_losses
Р	variables
єmetrics
Їlayers
їlayer_metrics
Э__call__
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses"
_generic_user_object
v
30
41
<2
=3
E4
F5
`6
a7
i8
j9
r10
s11"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╢
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
30
41"
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
<0
=1"
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
`0
a1"
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
i0
j1"
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
r0
s1"
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
Т2П
1__inference_Tree_Encoder_layer_call_fn_1294349193
1__inference_Tree_Encoder_layer_call_fn_1294350021
1__inference_Tree_Encoder_layer_call_fn_1294350102
1__inference_Tree_Encoder_layer_call_fn_1294349659└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
■2√
L__inference_Tree_Encoder_layer_call_and_return_conditional_losses_1294350321
L__inference_Tree_Encoder_layer_call_and_return_conditional_losses_1294350638
L__inference_Tree_Encoder_layer_call_and_return_conditional_losses_1294349758
L__inference_Tree_Encoder_layer_call_and_return_conditional_losses_1294349857└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ё2э
%__inference__wrapped_model_1294347835├
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *3в0
.К+
Inputs                  А
к2з
6__inference_Embedding_Dropout_layer_call_fn_1294350643
6__inference_Embedding_Dropout_layer_call_fn_1294350648┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
р2▌
Q__inference_Embedding_Dropout_layer_call_and_return_conditional_losses_1294350653
Q__inference_Embedding_Dropout_layer_call_and_return_conditional_losses_1294350665┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
█2╪
1__inference_Conv1D_3_128_layer_call_fn_1294350674в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ў2є
L__inference_Conv1D_3_128_layer_call_and_return_conditional_losses_1294350695в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
█2╪
1__inference_Conv1D_5_128_layer_call_fn_1294350704в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ў2є
L__inference_Conv1D_5_128_layer_call_and_return_conditional_losses_1294350725в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
█2╪
1__inference_Conv1D_7_128_layer_call_fn_1294350734в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ў2є
L__inference_Conv1D_7_128_layer_call_and_return_conditional_losses_1294350755в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2е
5__inference_Batch_Norm_3_128_layer_call_fn_1294350768
5__inference_Batch_Norm_3_128_layer_call_fn_1294350781┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
▐2█
P__inference_Batch_Norm_3_128_layer_call_and_return_conditional_losses_1294350801
P__inference_Batch_Norm_3_128_layer_call_and_return_conditional_losses_1294350835┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
и2е
5__inference_Batch_Norm_5_128_layer_call_fn_1294350848
5__inference_Batch_Norm_5_128_layer_call_fn_1294350861┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
▐2█
P__inference_Batch_Norm_5_128_layer_call_and_return_conditional_losses_1294350881
P__inference_Batch_Norm_5_128_layer_call_and_return_conditional_losses_1294350915┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
и2е
5__inference_Batch_Norm_7_128_layer_call_fn_1294350928
5__inference_Batch_Norm_7_128_layer_call_fn_1294350941┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
▐2█
P__inference_Batch_Norm_7_128_layer_call_and_return_conditional_losses_1294350961
P__inference_Batch_Norm_7_128_layer_call_and_return_conditional_losses_1294350995┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
█2╪
1__inference_Conv1D_3_256_layer_call_fn_1294351004в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ў2є
L__inference_Conv1D_3_256_layer_call_and_return_conditional_losses_1294351025в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
█2╪
1__inference_Conv1D_5_256_layer_call_fn_1294351034в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ў2є
L__inference_Conv1D_5_256_layer_call_and_return_conditional_losses_1294351055в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
█2╪
1__inference_Conv1D_7_256_layer_call_fn_1294351064в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ў2є
L__inference_Conv1D_7_256_layer_call_and_return_conditional_losses_1294351085в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2е
5__inference_Batch_Norm_3_256_layer_call_fn_1294351098
5__inference_Batch_Norm_3_256_layer_call_fn_1294351111┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
▐2█
P__inference_Batch_Norm_3_256_layer_call_and_return_conditional_losses_1294351131
P__inference_Batch_Norm_3_256_layer_call_and_return_conditional_losses_1294351165┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
и2е
5__inference_Batch_Norm_5_256_layer_call_fn_1294351178
5__inference_Batch_Norm_5_256_layer_call_fn_1294351191┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
▐2█
P__inference_Batch_Norm_5_256_layer_call_and_return_conditional_losses_1294351211
P__inference_Batch_Norm_5_256_layer_call_and_return_conditional_losses_1294351245┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
и2е
5__inference_Batch_Norm_7_256_layer_call_fn_1294351258
5__inference_Batch_Norm_7_256_layer_call_fn_1294351271┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
▐2█
P__inference_Batch_Norm_7_256_layer_call_and_return_conditional_losses_1294351291
P__inference_Batch_Norm_7_256_layer_call_and_return_conditional_losses_1294351325┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
К2З
/__inference_Max_Pool_3_layer_call_fn_1294348820╙
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *3в0
.К+'                           
е2в
J__inference_Max_Pool_3_layer_call_and_return_conditional_losses_1294348814╙
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *3в0
.К+'                           
К2З
/__inference_Max_Pool_5_layer_call_fn_1294348833╙
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *3в0
.К+'                           
е2в
J__inference_Max_Pool_5_layer_call_and_return_conditional_losses_1294348827╙
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *3в0
.К+'                           
К2З
/__inference_Max_Pool_7_layer_call_fn_1294348846╙
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *3в0
.К+'                           
е2в
J__inference_Max_Pool_7_layer_call_and_return_conditional_losses_1294348840╙
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *3в0
.К+'                           
▀2▄
5__inference_Pool_Concatenate_layer_call_fn_1294351332в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
·2ў
P__inference_Pool_Concatenate_layer_call_and_return_conditional_losses_1294351340в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
о2л
8__inference_Concatenate_Dropout_layer_call_fn_1294351345
8__inference_Concatenate_Dropout_layer_call_fn_1294351350┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ф2с
S__inference_Concatenate_Dropout_layer_call_and_return_conditional_losses_1294351355
S__inference_Concatenate_Dropout_layer_call_and_return_conditional_losses_1294351367┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
┘2╓
/__inference_Prediction_layer_call_fn_1294351376в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ї2ё
J__inference_Prediction_layer_call_and_return_conditional_losses_1294351387в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╬B╦
(__inference_signature_wrapper_1294349940Inputs"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 ╥
P__inference_Batch_Norm_3_128_layer_call_and_return_conditional_losses_1294350801~4132Aв>
7в4
.К+
inputs                  А
p 
к "3в0
)К&
0                  А
Ъ ╥
P__inference_Batch_Norm_3_128_layer_call_and_return_conditional_losses_1294350835~3412Aв>
7в4
.К+
inputs                  А
p
к "3в0
)К&
0                  А
Ъ к
5__inference_Batch_Norm_3_128_layer_call_fn_1294350768q4132Aв>
7в4
.К+
inputs                  А
p 
к "&К#                  Ак
5__inference_Batch_Norm_3_128_layer_call_fn_1294350781q3412Aв>
7в4
.К+
inputs                  А
p
к "&К#                  А╥
P__inference_Batch_Norm_3_256_layer_call_and_return_conditional_losses_1294351131~a^`_Aв>
7в4
.К+
inputs                  А
p 
к "3в0
)К&
0                  А
Ъ ╥
P__inference_Batch_Norm_3_256_layer_call_and_return_conditional_losses_1294351165~`a^_Aв>
7в4
.К+
inputs                  А
p
к "3в0
)К&
0                  А
Ъ к
5__inference_Batch_Norm_3_256_layer_call_fn_1294351098qa^`_Aв>
7в4
.К+
inputs                  А
p 
к "&К#                  Ак
5__inference_Batch_Norm_3_256_layer_call_fn_1294351111q`a^_Aв>
7в4
.К+
inputs                  А
p
к "&К#                  А╥
P__inference_Batch_Norm_5_128_layer_call_and_return_conditional_losses_1294350881~=:<;Aв>
7в4
.К+
inputs                  А
p 
к "3в0
)К&
0                  А
Ъ ╥
P__inference_Batch_Norm_5_128_layer_call_and_return_conditional_losses_1294350915~<=:;Aв>
7в4
.К+
inputs                  А
p
к "3в0
)К&
0                  А
Ъ к
5__inference_Batch_Norm_5_128_layer_call_fn_1294350848q=:<;Aв>
7в4
.К+
inputs                  А
p 
к "&К#                  Ак
5__inference_Batch_Norm_5_128_layer_call_fn_1294350861q<=:;Aв>
7в4
.К+
inputs                  А
p
к "&К#                  А╥
P__inference_Batch_Norm_5_256_layer_call_and_return_conditional_losses_1294351211~jgihAв>
7в4
.К+
inputs                  А
p 
к "3в0
)К&
0                  А
Ъ ╥
P__inference_Batch_Norm_5_256_layer_call_and_return_conditional_losses_1294351245~ijghAв>
7в4
.К+
inputs                  А
p
к "3в0
)К&
0                  А
Ъ к
5__inference_Batch_Norm_5_256_layer_call_fn_1294351178qjgihAв>
7в4
.К+
inputs                  А
p 
к "&К#                  Ак
5__inference_Batch_Norm_5_256_layer_call_fn_1294351191qijghAв>
7в4
.К+
inputs                  А
p
к "&К#                  А╥
P__inference_Batch_Norm_7_128_layer_call_and_return_conditional_losses_1294350961~FCEDAв>
7в4
.К+
inputs                  А
p 
к "3в0
)К&
0                  А
Ъ ╥
P__inference_Batch_Norm_7_128_layer_call_and_return_conditional_losses_1294350995~EFCDAв>
7в4
.К+
inputs                  А
p
к "3в0
)К&
0                  А
Ъ к
5__inference_Batch_Norm_7_128_layer_call_fn_1294350928qFCEDAв>
7в4
.К+
inputs                  А
p 
к "&К#                  Ак
5__inference_Batch_Norm_7_128_layer_call_fn_1294350941qEFCDAв>
7в4
.К+
inputs                  А
p
к "&К#                  А╥
P__inference_Batch_Norm_7_256_layer_call_and_return_conditional_losses_1294351291~sprqAв>
7в4
.К+
inputs                  А
p 
к "3в0
)К&
0                  А
Ъ ╥
P__inference_Batch_Norm_7_256_layer_call_and_return_conditional_losses_1294351325~rspqAв>
7в4
.К+
inputs                  А
p
к "3в0
)К&
0                  А
Ъ к
5__inference_Batch_Norm_7_256_layer_call_fn_1294351258qsprqAв>
7в4
.К+
inputs                  А
p 
к "&К#                  Ак
5__inference_Batch_Norm_7_256_layer_call_fn_1294351271qrspqAв>
7в4
.К+
inputs                  А
p
к "&К#                  А╡
S__inference_Concatenate_Dropout_layer_call_and_return_conditional_losses_1294351355^4в1
*в'
!К
inputs         А
p 
к "&в#
К
0         А
Ъ ╡
S__inference_Concatenate_Dropout_layer_call_and_return_conditional_losses_1294351367^4в1
*в'
!К
inputs         А
p
к "&в#
К
0         А
Ъ Н
8__inference_Concatenate_Dropout_layer_call_fn_1294351345Q4в1
*в'
!К
inputs         А
p 
к "К         АН
8__inference_Concatenate_Dropout_layer_call_fn_1294351350Q4в1
*в'
!К
inputs         А
p
к "К         А╚
L__inference_Conv1D_3_128_layer_call_and_return_conditional_losses_1294350695x=в:
3в0
.К+
inputs                  А
к "3в0
)К&
0                  А
Ъ а
1__inference_Conv1D_3_128_layer_call_fn_1294350674k=в:
3в0
.К+
inputs                  А
к "&К#                  А╚
L__inference_Conv1D_3_256_layer_call_and_return_conditional_losses_1294351025xKL=в:
3в0
.К+
inputs                  А
к "3в0
)К&
0                  А
Ъ а
1__inference_Conv1D_3_256_layer_call_fn_1294351004kKL=в:
3в0
.К+
inputs                  А
к "&К#                  А╚
L__inference_Conv1D_5_128_layer_call_and_return_conditional_losses_1294350725x$%=в:
3в0
.К+
inputs                  А
к "3в0
)К&
0                  А
Ъ а
1__inference_Conv1D_5_128_layer_call_fn_1294350704k$%=в:
3в0
.К+
inputs                  А
к "&К#                  А╚
L__inference_Conv1D_5_256_layer_call_and_return_conditional_losses_1294351055xQR=в:
3в0
.К+
inputs                  А
к "3в0
)К&
0                  А
Ъ а
1__inference_Conv1D_5_256_layer_call_fn_1294351034kQR=в:
3в0
.К+
inputs                  А
к "&К#                  А╚
L__inference_Conv1D_7_128_layer_call_and_return_conditional_losses_1294350755x*+=в:
3в0
.К+
inputs                  А
к "3в0
)К&
0                  А
Ъ а
1__inference_Conv1D_7_128_layer_call_fn_1294350734k*+=в:
3в0
.К+
inputs                  А
к "&К#                  А╚
L__inference_Conv1D_7_256_layer_call_and_return_conditional_losses_1294351085xWX=в:
3в0
.К+
inputs                  А
к "3в0
)К&
0                  А
Ъ а
1__inference_Conv1D_7_256_layer_call_fn_1294351064kWX=в:
3в0
.К+
inputs                  А
к "&К#                  А═
Q__inference_Embedding_Dropout_layer_call_and_return_conditional_losses_1294350653xAв>
7в4
.К+
inputs                  А
p 
к "3в0
)К&
0                  А
Ъ ═
Q__inference_Embedding_Dropout_layer_call_and_return_conditional_losses_1294350665xAв>
7в4
.К+
inputs                  А
p
к "3в0
)К&
0                  А
Ъ е
6__inference_Embedding_Dropout_layer_call_fn_1294350643kAв>
7в4
.К+
inputs                  А
p 
к "&К#                  Ае
6__inference_Embedding_Dropout_layer_call_fn_1294350648kAв>
7в4
.К+
inputs                  А
p
к "&К#                  А┼
J__inference_Max_Pool_3_layer_call_and_return_conditional_losses_1294348814wEвB
;в8
6К3
inputs'                           
к ".в+
$К!
0                  
Ъ Э
/__inference_Max_Pool_3_layer_call_fn_1294348820jEвB
;в8
6К3
inputs'                           
к "!К                  ┼
J__inference_Max_Pool_5_layer_call_and_return_conditional_losses_1294348827wEвB
;в8
6К3
inputs'                           
к ".в+
$К!
0                  
Ъ Э
/__inference_Max_Pool_5_layer_call_fn_1294348833jEвB
;в8
6К3
inputs'                           
к "!К                  ┼
J__inference_Max_Pool_7_layer_call_and_return_conditional_losses_1294348840wEвB
;в8
6К3
inputs'                           
к ".в+
$К!
0                  
Ъ Э
/__inference_Max_Pool_7_layer_call_fn_1294348846jEвB
;в8
6К3
inputs'                           
к "!К                  Б
P__inference_Pool_Concatenate_layer_call_and_return_conditional_losses_1294351340мБв~
wвt
rЪo
#К 
inputs/0         А
#К 
inputs/1         А
#К 
inputs/2         А
к "&в#
К
0         А
Ъ ┘
5__inference_Pool_Concatenate_layer_call_fn_1294351332ЯБв~
wвt
rЪo
#К 
inputs/0         А
#К 
inputs/1         А
#К 
inputs/2         А
к "К         Ан
J__inference_Prediction_layer_call_and_return_conditional_losses_1294351387_МН0в-
&в#
!К
inputs         А
к "%в"
К
0         
Ъ Е
/__inference_Prediction_layer_call_fn_1294351376RМН0в-
&в#
!К
inputs         А
к "К         щ
L__inference_Tree_Encoder_layer_call_and_return_conditional_losses_1294349758Ш(*+$%FCED=:<;4132WXQRKLsprqjgiha^`_МНEвB
;в8
.К+
Inputs                  А
p 

 
к "%в"
К
0         
Ъ щ
L__inference_Tree_Encoder_layer_call_and_return_conditional_losses_1294349857Ш(*+$%EFCD<=:;3412WXQRKLrspqijgh`a^_МНEвB
;в8
.К+
Inputs                  А
p

 
к "%в"
К
0         
Ъ щ
L__inference_Tree_Encoder_layer_call_and_return_conditional_losses_1294350321Ш(*+$%FCED=:<;4132WXQRKLsprqjgiha^`_МНEвB
;в8
.К+
inputs                  А
p 

 
к "%в"
К
0         
Ъ щ
L__inference_Tree_Encoder_layer_call_and_return_conditional_losses_1294350638Ш(*+$%EFCD<=:;3412WXQRKLrspqijgh`a^_МНEвB
;в8
.К+
inputs                  А
p

 
к "%в"
К
0         
Ъ ┴
1__inference_Tree_Encoder_layer_call_fn_1294349193Л(*+$%FCED=:<;4132WXQRKLsprqjgiha^`_МНEвB
;в8
.К+
Inputs                  А
p 

 
к "К         ┴
1__inference_Tree_Encoder_layer_call_fn_1294349659Л(*+$%EFCD<=:;3412WXQRKLrspqijgh`a^_МНEвB
;в8
.К+
Inputs                  А
p

 
к "К         ┴
1__inference_Tree_Encoder_layer_call_fn_1294350021Л(*+$%FCED=:<;4132WXQRKLsprqjgiha^`_МНEвB
;в8
.К+
inputs                  А
p 

 
к "К         ┴
1__inference_Tree_Encoder_layer_call_fn_1294350102Л(*+$%EFCD<=:;3412WXQRKLrspqijgh`a^_МНEвB
;в8
.К+
inputs                  А
p

 
к "К         ╠
%__inference__wrapped_model_1294347835в(*+$%FCED=:<;4132WXQRKLsprqjgiha^`_МН=в:
3в0
.К+
Inputs                  А
к "7к4
2

Prediction$К!

Prediction         ┘
(__inference_signature_wrapper_1294349940м(*+$%FCED=:<;4132WXQRKLsprqjgiha^`_МНGвD
в 
=к:
8
Inputs.К+
Inputs                  А"7к4
2

Prediction$К!

Prediction         