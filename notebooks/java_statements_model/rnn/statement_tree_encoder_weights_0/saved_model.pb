ð/
ë
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
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
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
l
	ReverseV2
tensor"T
axis"Tidx
output"T"
Tidxtype0:
2	"
Ttype:
2	

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
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
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
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype

SplitV

value"T
size_splits"Tlen
	split_dim
output"T*	num_split"
	num_splitint(0"	
Ttype"
Tlentype0	:
2	
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
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
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
-
Tanh
x"T
y"T"
Ttype:

2

TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type
output_handle"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements

handle"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintÿÿÿÿÿÿÿÿÿ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718é,

Prediction/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*"
shared_namePrediction/kernel
x
%Prediction/kernel/Read/ReadVariableOpReadVariableOpPrediction/kernel*
_output_shapes
:	*
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
Ò
:Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*K
shared_name<:Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/kernel
Ë
NDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/kernel/Read/ReadVariableOpReadVariableOp:Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/kernel* 
_output_shapes
:
*
dtype0
æ
DDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*U
shared_nameFDDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/recurrent_kernel
ß
XDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/recurrent_kernel/Read/ReadVariableOpReadVariableOpDDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/recurrent_kernel* 
_output_shapes
:
*
dtype0
Í
8Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*I
shared_name:8Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/bias
Æ
LDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/bias/Read/ReadVariableOpReadVariableOp8Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/bias*
_output_shapes
:	*
dtype0
Ô
;Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*L
shared_name=;Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/kernel
Í
ODouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/kernel/Read/ReadVariableOpReadVariableOp;Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/kernel* 
_output_shapes
:
*
dtype0
è
EDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*V
shared_nameGEDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/recurrent_kernel
á
YDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/recurrent_kernel/Read/ReadVariableOpReadVariableOpEDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/recurrent_kernel* 
_output_shapes
:
*
dtype0
Ï
9Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*J
shared_name;9Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/bias
È
MDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/bias/Read/ReadVariableOpReadVariableOp9Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/bias*
_output_shapes
:	*
dtype0

NoOpNoOp
þ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*¹
value¯B¬ B¥
Ê
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
trainable_variables
regularization_losses
	variables
	keras_api
	
signatures
 
R

trainable_variables
regularization_losses
	variables
	keras_api
y
forward_layer
backward_layer
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
8
0
1
2
3
4
5
6
7
 
8
0
1
2
3
4
5
6
7
­
trainable_variables
 non_trainable_variables
!layer_regularization_losses
regularization_losses
	variables
"metrics

#layers
$layer_metrics
 
 
 
 
­

trainable_variables
%non_trainable_variables
&layer_regularization_losses
regularization_losses
	variables
'metrics

(layers
)layer_metrics
l
*cell
+
state_spec
,trainable_variables
-regularization_losses
.	variables
/	keras_api
l
0cell
1
state_spec
2trainable_variables
3regularization_losses
4	variables
5	keras_api
*
0
1
2
3
4
5
 
*
0
1
2
3
4
5
­
trainable_variables
6non_trainable_variables
7layer_regularization_losses
regularization_losses
	variables
8metrics

9layers
:layer_metrics
][
VARIABLE_VALUEPrediction/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEPrediction/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
trainable_variables
;non_trainable_variables
<layer_regularization_losses
regularization_losses
	variables
=metrics

>layers
?layer_metrics
~
VARIABLE_VALUE:Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEDDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE8Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE;Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/kernel0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEEDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/recurrent_kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE9Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

0
1
2
3
 
 
 
 
 
 
~

kernel
recurrent_kernel
bias
@trainable_variables
Aregularization_losses
B	variables
C	keras_api
 

0
1
2
 

0
1
2
¹
,trainable_variables
Dnon_trainable_variables
Elayer_regularization_losses
-regularization_losses
.	variables
Fmetrics

Glayers
Hlayer_metrics

Istates
~

kernel
recurrent_kernel
bias
Jtrainable_variables
Kregularization_losses
L	variables
M	keras_api
 

0
1
2
 

0
1
2
¹
2trainable_variables
Nnon_trainable_variables
Olayer_regularization_losses
3regularization_losses
4	variables
Pmetrics

Qlayers
Rlayer_metrics

Sstates
 
 
 

0
1
 
 
 
 
 
 

0
1
2
 

0
1
2
­
@trainable_variables
Tnon_trainable_variables
Ulayer_regularization_losses
Aregularization_losses
B	variables
Vmetrics

Wlayers
Xlayer_metrics
 
 
 

*0
 
 

0
1
2
 

0
1
2
­
Jtrainable_variables
Ynon_trainable_variables
Zlayer_regularization_losses
Kregularization_losses
L	variables
[metrics

\layers
]layer_metrics
 
 
 

00
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

serving_default_InputsPlaceholder*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype0**
shape!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
æ
StatefulPartitionedCallStatefulPartitionedCallserving_default_Inputs8Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/bias:Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/kernelDDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/recurrent_kernel9Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/bias;Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/kernelEDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/recurrent_kernelPrediction/kernelPrediction/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *0
f+R)
'__inference_signature_wrapper_387380633
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
è
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%Prediction/kernel/Read/ReadVariableOp#Prediction/bias/Read/ReadVariableOpNDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/kernel/Read/ReadVariableOpXDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/recurrent_kernel/Read/ReadVariableOpLDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/bias/Read/ReadVariableOpODouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/kernel/Read/ReadVariableOpYDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/recurrent_kernel/Read/ReadVariableOpMDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/bias/Read/ReadVariableOpConst*
Tin
2
*
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
GPU2*0J 8 *+
f&R$
"__inference__traced_save_387382958
Ã
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamePrediction/kernelPrediction/bias:Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/kernelDDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/recurrent_kernel8Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/bias;Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/kernelEDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/recurrent_kernel9Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/bias*
Tin
2	*
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
GPU2*0J 8 *.
f)R'
%__inference__traced_restore_387382992í,
=

M__inference_forward_rnn_25_layer_call_and_return_conditional_losses_387378707

inputs(
gru_cell_25_387378632:	)
gru_cell_25_387378634:
)
gru_cell_25_387378636:

identity¢#gru_cell_25/StatefulPartitionedCall¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2
#gru_cell_25/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_25_387378632gru_cell_25_387378634gru_cell_25_387378636*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_gru_cell_25_layer_call_and_return_conditional_losses_3873786312%
#gru_cell_25/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterø
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_25_387378632gru_cell_25_387378634gru_cell_25_387378636*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	* 
bodyR
while_body_387378644* 
condR
while_cond_387378643*9
output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
transpose_1
IdentityIdentitystrided_slice_3:output:0$^gru_cell_25/StatefulPartitionedCall^while*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2J
#gru_cell_25/StatefulPartitionedCall#gru_cell_25/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¤
á
=Double_Bidirectional_GRU_backward_rnn_25_while_body_387380898n
jdouble_bidirectional_gru_backward_rnn_25_while_double_bidirectional_gru_backward_rnn_25_while_loop_countert
pdouble_bidirectional_gru_backward_rnn_25_while_double_bidirectional_gru_backward_rnn_25_while_maximum_iterations>
:double_bidirectional_gru_backward_rnn_25_while_placeholder@
<double_bidirectional_gru_backward_rnn_25_while_placeholder_1@
<double_bidirectional_gru_backward_rnn_25_while_placeholder_2m
idouble_bidirectional_gru_backward_rnn_25_while_double_bidirectional_gru_backward_rnn_25_strided_slice_1_0ª
¥double_bidirectional_gru_backward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_double_bidirectional_gru_backward_rnn_25_tensorarrayunstack_tensorlistfromtensor_0g
Tdouble_bidirectional_gru_backward_rnn_25_while_gru_cell_25_readvariableop_resource_0:	o
[double_bidirectional_gru_backward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource_0:
q
]double_bidirectional_gru_backward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource_0:
;
7double_bidirectional_gru_backward_rnn_25_while_identity=
9double_bidirectional_gru_backward_rnn_25_while_identity_1=
9double_bidirectional_gru_backward_rnn_25_while_identity_2=
9double_bidirectional_gru_backward_rnn_25_while_identity_3=
9double_bidirectional_gru_backward_rnn_25_while_identity_4k
gdouble_bidirectional_gru_backward_rnn_25_while_double_bidirectional_gru_backward_rnn_25_strided_slice_1¨
£double_bidirectional_gru_backward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_double_bidirectional_gru_backward_rnn_25_tensorarrayunstack_tensorlistfromtensore
Rdouble_bidirectional_gru_backward_rnn_25_while_gru_cell_25_readvariableop_resource:	m
Ydouble_bidirectional_gru_backward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource:
o
[double_bidirectional_gru_backward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource:
¢PDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp¢RDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp¢IDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/ReadVariableOp
`Double_Bidirectional_GRU/backward_rnn_25/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2b
`Double_Bidirectional_GRU/backward_rnn_25/while/TensorArrayV2Read/TensorListGetItem/element_shapeË
RDouble_Bidirectional_GRU/backward_rnn_25/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem¥double_bidirectional_gru_backward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_double_bidirectional_gru_backward_rnn_25_tensorarrayunstack_tensorlistfromtensor_0:double_bidirectional_gru_backward_rnn_25_while_placeholderiDouble_Bidirectional_GRU/backward_rnn_25/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02T
RDouble_Bidirectional_GRU/backward_rnn_25/while/TensorArrayV2Read/TensorListGetItem¬
IDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/ReadVariableOpReadVariableOpTdouble_bidirectional_gru_backward_rnn_25_while_gru_cell_25_readvariableop_resource_0*
_output_shapes
:	*
dtype02K
IDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/ReadVariableOp
BDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/unstackUnpackQDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2D
BDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/unstackÂ
PDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOpReadVariableOp[double_bidirectional_gru_backward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype02R
PDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOpø
ADouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMulMatMulYDouble_Bidirectional_GRU/backward_rnn_25/while/TensorArrayV2Read/TensorListGetItem:item:0XDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2C
ADouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMulà
BDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/BiasAddBiasAddKDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul:product:0KDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2D
BDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/BiasAddã
JDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2L
JDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/split/split_dim
@Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/splitSplitSDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/split/split_dim:output:0KDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2B
@Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/splitÈ
RDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpReadVariableOp]double_bidirectional_gru_backward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02T
RDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpá
CDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul_1MatMul<double_bidirectional_gru_backward_rnn_25_while_placeholder_2ZDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2E
CDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul_1æ
DDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/BiasAdd_1BiasAddMDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul_1:product:0KDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2F
DDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/BiasAdd_1Ù
@Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2B
@Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/Constç
LDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2N
LDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/split_1/split_dimû
BDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/split_1SplitVMDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/BiasAdd_1:output:0IDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/Const:output:0UDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2D
BDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/split_1Ô
>Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/addAddV2IDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/split:output:0KDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2@
>Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/add
BDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/SigmoidSigmoidBDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2D
BDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/SigmoidØ
@Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/add_1AddV2IDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/split:output:1KDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2B
@Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/add_1
DDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/Sigmoid_1SigmoidDDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2F
DDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/Sigmoid_1Ñ
>Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/mulMulHDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/Sigmoid_1:y:0KDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2@
>Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/mulÏ
@Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/add_2AddV2IDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/split:output:2BDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2B
@Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/add_2
?Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/TanhTanhDDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2A
?Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/TanhÄ
@Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/mul_1MulFDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/Sigmoid:y:0<double_bidirectional_gru_backward_rnn_25_while_placeholder_2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2B
@Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/mul_1É
@Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2B
@Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/sub/xÍ
>Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/subSubIDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/sub/x:output:0FDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2@
>Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/subÇ
@Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/mul_2MulBDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/sub:z:0CDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2B
@Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/mul_2Ì
@Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/add_3AddV2DDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/mul_1:z:0DDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2B
@Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/add_3¬
SDouble_Bidirectional_GRU/backward_rnn_25/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem<double_bidirectional_gru_backward_rnn_25_while_placeholder_1:double_bidirectional_gru_backward_rnn_25_while_placeholderDDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/add_3:z:0*
_output_shapes
: *
element_dtype02U
SDouble_Bidirectional_GRU/backward_rnn_25/while/TensorArrayV2Write/TensorListSetItem®
4Double_Bidirectional_GRU/backward_rnn_25/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :26
4Double_Bidirectional_GRU/backward_rnn_25/while/add/y
2Double_Bidirectional_GRU/backward_rnn_25/while/addAddV2:double_bidirectional_gru_backward_rnn_25_while_placeholder=Double_Bidirectional_GRU/backward_rnn_25/while/add/y:output:0*
T0*
_output_shapes
: 24
2Double_Bidirectional_GRU/backward_rnn_25/while/add²
6Double_Bidirectional_GRU/backward_rnn_25/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :28
6Double_Bidirectional_GRU/backward_rnn_25/while/add_1/yÃ
4Double_Bidirectional_GRU/backward_rnn_25/while/add_1AddV2jdouble_bidirectional_gru_backward_rnn_25_while_double_bidirectional_gru_backward_rnn_25_while_loop_counter?Double_Bidirectional_GRU/backward_rnn_25/while/add_1/y:output:0*
T0*
_output_shapes
: 26
4Double_Bidirectional_GRU/backward_rnn_25/while/add_1Í
7Double_Bidirectional_GRU/backward_rnn_25/while/IdentityIdentity8Double_Bidirectional_GRU/backward_rnn_25/while/add_1:z:0Q^Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOpS^Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpJ^Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 29
7Double_Bidirectional_GRU/backward_rnn_25/while/Identity
9Double_Bidirectional_GRU/backward_rnn_25/while/Identity_1Identitypdouble_bidirectional_gru_backward_rnn_25_while_double_bidirectional_gru_backward_rnn_25_while_maximum_iterationsQ^Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOpS^Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpJ^Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2;
9Double_Bidirectional_GRU/backward_rnn_25/while/Identity_1Ï
9Double_Bidirectional_GRU/backward_rnn_25/while/Identity_2Identity6Double_Bidirectional_GRU/backward_rnn_25/while/add:z:0Q^Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOpS^Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpJ^Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2;
9Double_Bidirectional_GRU/backward_rnn_25/while/Identity_2ü
9Double_Bidirectional_GRU/backward_rnn_25/while/Identity_3IdentitycDouble_Bidirectional_GRU/backward_rnn_25/while/TensorArrayV2Write/TensorListSetItem:output_handle:0Q^Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOpS^Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpJ^Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2;
9Double_Bidirectional_GRU/backward_rnn_25/while/Identity_3ï
9Double_Bidirectional_GRU/backward_rnn_25/while/Identity_4IdentityDDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/add_3:z:0Q^Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOpS^Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpJ^Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2;
9Double_Bidirectional_GRU/backward_rnn_25/while/Identity_4"Ô
gdouble_bidirectional_gru_backward_rnn_25_while_double_bidirectional_gru_backward_rnn_25_strided_slice_1idouble_bidirectional_gru_backward_rnn_25_while_double_bidirectional_gru_backward_rnn_25_strided_slice_1_0"¼
[double_bidirectional_gru_backward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource]double_bidirectional_gru_backward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource_0"¸
Ydouble_bidirectional_gru_backward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource[double_bidirectional_gru_backward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource_0"ª
Rdouble_bidirectional_gru_backward_rnn_25_while_gru_cell_25_readvariableop_resourceTdouble_bidirectional_gru_backward_rnn_25_while_gru_cell_25_readvariableop_resource_0"{
7double_bidirectional_gru_backward_rnn_25_while_identity@Double_Bidirectional_GRU/backward_rnn_25/while/Identity:output:0"
9double_bidirectional_gru_backward_rnn_25_while_identity_1BDouble_Bidirectional_GRU/backward_rnn_25/while/Identity_1:output:0"
9double_bidirectional_gru_backward_rnn_25_while_identity_2BDouble_Bidirectional_GRU/backward_rnn_25/while/Identity_2:output:0"
9double_bidirectional_gru_backward_rnn_25_while_identity_3BDouble_Bidirectional_GRU/backward_rnn_25/while/Identity_3:output:0"
9double_bidirectional_gru_backward_rnn_25_while_identity_4BDouble_Bidirectional_GRU/backward_rnn_25/while/Identity_4:output:0"Î
£double_bidirectional_gru_backward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_double_bidirectional_gru_backward_rnn_25_tensorarrayunstack_tensorlistfromtensor¥double_bidirectional_gru_backward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_double_bidirectional_gru_backward_rnn_25_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : 2¤
PDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOpPDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp2¨
RDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpRDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp2
IDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/ReadVariableOpIDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 

¶
ITree_Encoder_Double_Bidirectional_GRU_forward_rnn_25_while_cond_387378305
tree_encoder_double_bidirectional_gru_forward_rnn_25_while_tree_encoder_double_bidirectional_gru_forward_rnn_25_while_loop_counter
tree_encoder_double_bidirectional_gru_forward_rnn_25_while_tree_encoder_double_bidirectional_gru_forward_rnn_25_while_maximum_iterationsJ
Ftree_encoder_double_bidirectional_gru_forward_rnn_25_while_placeholderL
Htree_encoder_double_bidirectional_gru_forward_rnn_25_while_placeholder_1L
Htree_encoder_double_bidirectional_gru_forward_rnn_25_while_placeholder_2
tree_encoder_double_bidirectional_gru_forward_rnn_25_while_less_tree_encoder_double_bidirectional_gru_forward_rnn_25_strided_slice_1¢
tree_encoder_double_bidirectional_gru_forward_rnn_25_while_tree_encoder_double_bidirectional_gru_forward_rnn_25_while_cond_387378305___redundant_placeholder0¢
tree_encoder_double_bidirectional_gru_forward_rnn_25_while_tree_encoder_double_bidirectional_gru_forward_rnn_25_while_cond_387378305___redundant_placeholder1¢
tree_encoder_double_bidirectional_gru_forward_rnn_25_while_tree_encoder_double_bidirectional_gru_forward_rnn_25_while_cond_387378305___redundant_placeholder2¢
tree_encoder_double_bidirectional_gru_forward_rnn_25_while_tree_encoder_double_bidirectional_gru_forward_rnn_25_while_cond_387378305___redundant_placeholder3G
Ctree_encoder_double_bidirectional_gru_forward_rnn_25_while_identity
ú
?Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/LessLessFtree_encoder_double_bidirectional_gru_forward_rnn_25_while_placeholdertree_encoder_double_bidirectional_gru_forward_rnn_25_while_less_tree_encoder_double_bidirectional_gru_forward_rnn_25_strided_slice_1*
T0*
_output_shapes
: 2A
?Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/Lessü
CTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/IdentityIdentityCTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/Less:z:0*
T0
*
_output_shapes
: 2E
CTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/Identity"
Ctree_encoder_double_bidirectional_gru_forward_rnn_25_while_identityLTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
î
Ý
K__inference_Tree_Encoder_layer_call_and_return_conditional_losses_387380587

inputs5
"double_bidirectional_gru_387380568:	6
"double_bidirectional_gru_387380570:
6
"double_bidirectional_gru_387380572:
5
"double_bidirectional_gru_387380574:	6
"double_bidirectional_gru_387380576:
6
"double_bidirectional_gru_387380578:
'
prediction_387380581:	"
prediction_387380583:
identity¢0Double_Bidirectional_GRU/StatefulPartitionedCall¢"Prediction/StatefulPartitionedCall
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
GPU2*0J 8 *Y
fTRR
P__inference_Embedding_Dropout_layer_call_and_return_conditional_losses_3873797332#
!Embedding_Dropout/PartitionedCall­
0Double_Bidirectional_GRU/StatefulPartitionedCallStatefulPartitionedCall*Embedding_Dropout/PartitionedCall:output:0"double_bidirectional_gru_387380568"double_bidirectional_gru_387380570"double_bidirectional_gru_387380572"double_bidirectional_gru_387380574"double_bidirectional_gru_387380576"double_bidirectional_gru_387380578*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *`
f[RY
W__inference_Double_Bidirectional_GRU_layer_call_and_return_conditional_losses_38738004722
0Double_Bidirectional_GRU/StatefulPartitionedCallÝ
"Prediction/StatefulPartitionedCallStatefulPartitionedCall9Double_Bidirectional_GRU/StatefulPartitionedCall:output:0prediction_387380581prediction_387380583*
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
GPU2*0J 8 *R
fMRK
I__inference_Prediction_layer_call_and_return_conditional_losses_3873800722$
"Prediction/StatefulPartitionedCall×
IdentityIdentity+Prediction/StatefulPartitionedCall:output:01^Double_Bidirectional_GRU/StatefulPartitionedCall#^Prediction/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : 2d
0Double_Bidirectional_GRU/StatefulPartitionedCall0Double_Bidirectional_GRU/StatefulPartitionedCall2H
"Prediction/StatefulPartitionedCall"Prediction/StatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameInputs

¹
while_cond_387382272
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_17
3while_while_cond_387382272___redundant_placeholder07
3while_while_cond_387382272___redundant_placeholder17
3while_while_cond_387382272___redundant_placeholder27
3while_while_cond_387382272___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Û	
Ö
0__inference_Tree_Encoder_layer_call_fn_387380675

inputs
unknown:	
	unknown_0:

	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:

	unknown_5:	
	unknown_6:
identity¢StatefulPartitionedCallÌ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_Tree_Encoder_layer_call_and_return_conditional_losses_3873805242
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
À
Ö
#forward_rnn_25_while_cond_387380195:
6forward_rnn_25_while_forward_rnn_25_while_loop_counter@
<forward_rnn_25_while_forward_rnn_25_while_maximum_iterations$
 forward_rnn_25_while_placeholder&
"forward_rnn_25_while_placeholder_1&
"forward_rnn_25_while_placeholder_2<
8forward_rnn_25_while_less_forward_rnn_25_strided_slice_1U
Qforward_rnn_25_while_forward_rnn_25_while_cond_387380195___redundant_placeholder0U
Qforward_rnn_25_while_forward_rnn_25_while_cond_387380195___redundant_placeholder1U
Qforward_rnn_25_while_forward_rnn_25_while_cond_387380195___redundant_placeholder2U
Qforward_rnn_25_while_forward_rnn_25_while_cond_387380195___redundant_placeholder3!
forward_rnn_25_while_identity
»
forward_rnn_25/while/LessLess forward_rnn_25_while_placeholder8forward_rnn_25_while_less_forward_rnn_25_strided_slice_1*
T0*
_output_shapes
: 2
forward_rnn_25/while/Less
forward_rnn_25/while/IdentityIdentityforward_rnn_25/while/Less:z:0*
T0
*
_output_shapes
: 2
forward_rnn_25/while/Identity"G
forward_rnn_25_while_identity&forward_rnn_25/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ö\

N__inference_backward_rnn_25_layer_call_and_return_conditional_losses_387382699
inputs_06
#gru_cell_25_readvariableop_resource:	>
*gru_cell_25_matmul_readvariableop_resource:
@
,gru_cell_25_matmul_1_readvariableop_resource:

identity¢!gru_cell_25/MatMul/ReadVariableOp¢#gru_cell_25/MatMul_1/ReadVariableOp¢gru_cell_25/ReadVariableOp¢whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2j
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2
ReverseV2/axis
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	ReverseV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeý
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2
gru_cell_25/ReadVariableOpReadVariableOp#gru_cell_25_readvariableop_resource*
_output_shapes
:	*
dtype02
gru_cell_25/ReadVariableOp
gru_cell_25/unstackUnpack"gru_cell_25/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
gru_cell_25/unstack³
!gru_cell_25/MatMul/ReadVariableOpReadVariableOp*gru_cell_25_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02#
!gru_cell_25/MatMul/ReadVariableOpª
gru_cell_25/MatMulMatMulstrided_slice_2:output:0)gru_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/MatMul¤
gru_cell_25/BiasAddBiasAddgru_cell_25/MatMul:product:0gru_cell_25/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/BiasAdd
gru_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/split/split_dimß
gru_cell_25/splitSplit$gru_cell_25/split/split_dim:output:0gru_cell_25/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
gru_cell_25/split¹
#gru_cell_25/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_25_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02%
#gru_cell_25/MatMul_1/ReadVariableOp¦
gru_cell_25/MatMul_1MatMulzeros:output:0+gru_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/MatMul_1ª
gru_cell_25/BiasAdd_1BiasAddgru_cell_25/MatMul_1:product:0gru_cell_25/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/BiasAdd_1{
gru_cell_25/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2
gru_cell_25/Const
gru_cell_25/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/split_1/split_dim
gru_cell_25/split_1SplitVgru_cell_25/BiasAdd_1:output:0gru_cell_25/Const:output:0&gru_cell_25/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
gru_cell_25/split_1
gru_cell_25/addAddV2gru_cell_25/split:output:0gru_cell_25/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/add}
gru_cell_25/SigmoidSigmoidgru_cell_25/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/Sigmoid
gru_cell_25/add_1AddV2gru_cell_25/split:output:1gru_cell_25/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/add_1
gru_cell_25/Sigmoid_1Sigmoidgru_cell_25/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/Sigmoid_1
gru_cell_25/mulMulgru_cell_25/Sigmoid_1:y:0gru_cell_25/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/mul
gru_cell_25/add_2AddV2gru_cell_25/split:output:2gru_cell_25/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/add_2v
gru_cell_25/TanhTanhgru_cell_25/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/Tanh
gru_cell_25/mul_1Mulgru_cell_25/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/mul_1k
gru_cell_25/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_25/sub/x
gru_cell_25/subSubgru_cell_25/sub/x:output:0gru_cell_25/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/sub
gru_cell_25/mul_2Mulgru_cell_25/sub:z:0gru_cell_25/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/mul_2
gru_cell_25/add_3AddV2gru_cell_25/mul_1:z:0gru_cell_25/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter²
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_25_readvariableop_resource*gru_cell_25_matmul_readvariableop_resource,gru_cell_25_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	* 
bodyR
while_body_387382611* 
condR
while_cond_387382610*9
output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
transpose_1Ü
IdentityIdentitystrided_slice_3:output:0"^gru_cell_25/MatMul/ReadVariableOp$^gru_cell_25/MatMul_1/ReadVariableOp^gru_cell_25/ReadVariableOp^while*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2F
!gru_cell_25/MatMul/ReadVariableOp!gru_cell_25/MatMul/ReadVariableOp2J
#gru_cell_25/MatMul_1/ReadVariableOp#gru_cell_25/MatMul_1/ReadVariableOp28
gru_cell_25/ReadVariableOpgru_cell_25/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
©
o
P__inference_Embedding_Dropout_layer_call_and_return_conditional_losses_387380475

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
¬#
Î
while_body_387379224
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_gru_cell_25_387379246_0:	1
while_gru_cell_25_387379248_0:
1
while_gru_cell_25_387379250_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_gru_cell_25_387379246:	/
while_gru_cell_25_387379248:
/
while_gru_cell_25_387379250:
¢)while/gru_cell_25/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÂ
)while/gru_cell_25/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_25_387379246_0while_gru_cell_25_387379248_0while_gru_cell_25_387379250_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_gru_cell_25_layer_call_and_return_conditional_losses_3873792112+
)while/gru_cell_25/StatefulPartitionedCallö
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_25/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:0*^while/gru_cell_25/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations*^while/gru_cell_25/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0*^while/gru_cell_25/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¹
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/gru_cell_25/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3Ã
while/Identity_4Identity2while/gru_cell_25/StatefulPartitionedCall:output:1*^while/gru_cell_25/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4"<
while_gru_cell_25_387379246while_gru_cell_25_387379246_0"<
while_gru_cell_25_387379248while_gru_cell_25_387379248_0"<
while_gru_cell_25_387379250while_gru_cell_25_387379250_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/gru_cell_25/StatefulPartitionedCall)while/gru_cell_25/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Äa
ï
$backward_rnn_25_while_body_387380350<
8backward_rnn_25_while_backward_rnn_25_while_loop_counterB
>backward_rnn_25_while_backward_rnn_25_while_maximum_iterations%
!backward_rnn_25_while_placeholder'
#backward_rnn_25_while_placeholder_1'
#backward_rnn_25_while_placeholder_2;
7backward_rnn_25_while_backward_rnn_25_strided_slice_1_0w
sbackward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_backward_rnn_25_tensorarrayunstack_tensorlistfromtensor_0N
;backward_rnn_25_while_gru_cell_25_readvariableop_resource_0:	V
Bbackward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource_0:
X
Dbackward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource_0:
"
backward_rnn_25_while_identity$
 backward_rnn_25_while_identity_1$
 backward_rnn_25_while_identity_2$
 backward_rnn_25_while_identity_3$
 backward_rnn_25_while_identity_49
5backward_rnn_25_while_backward_rnn_25_strided_slice_1u
qbackward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_backward_rnn_25_tensorarrayunstack_tensorlistfromtensorL
9backward_rnn_25_while_gru_cell_25_readvariableop_resource:	T
@backward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource:
V
Bbackward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource:
¢7backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp¢9backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp¢0backward_rnn_25/while/gru_cell_25/ReadVariableOpã
Gbackward_rnn_25/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2I
Gbackward_rnn_25/while/TensorArrayV2Read/TensorListGetItem/element_shape´
9backward_rnn_25/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsbackward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_backward_rnn_25_tensorarrayunstack_tensorlistfromtensor_0!backward_rnn_25_while_placeholderPbackward_rnn_25/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02;
9backward_rnn_25/while/TensorArrayV2Read/TensorListGetItemá
0backward_rnn_25/while/gru_cell_25/ReadVariableOpReadVariableOp;backward_rnn_25_while_gru_cell_25_readvariableop_resource_0*
_output_shapes
:	*
dtype022
0backward_rnn_25/while/gru_cell_25/ReadVariableOpÒ
)backward_rnn_25/while/gru_cell_25/unstackUnpack8backward_rnn_25/while/gru_cell_25/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2+
)backward_rnn_25/while/gru_cell_25/unstack÷
7backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOpReadVariableOpBbackward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype029
7backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp
(backward_rnn_25/while/gru_cell_25/MatMulMatMul@backward_rnn_25/while/TensorArrayV2Read/TensorListGetItem:item:0?backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(backward_rnn_25/while/gru_cell_25/MatMulü
)backward_rnn_25/while/gru_cell_25/BiasAddBiasAdd2backward_rnn_25/while/gru_cell_25/MatMul:product:02backward_rnn_25/while/gru_cell_25/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)backward_rnn_25/while/gru_cell_25/BiasAdd±
1backward_rnn_25/while/gru_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ23
1backward_rnn_25/while/gru_cell_25/split/split_dim·
'backward_rnn_25/while/gru_cell_25/splitSplit:backward_rnn_25/while/gru_cell_25/split/split_dim:output:02backward_rnn_25/while/gru_cell_25/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2)
'backward_rnn_25/while/gru_cell_25/splitý
9backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpReadVariableOpDbackward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02;
9backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpý
*backward_rnn_25/while/gru_cell_25/MatMul_1MatMul#backward_rnn_25_while_placeholder_2Abackward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*backward_rnn_25/while/gru_cell_25/MatMul_1
+backward_rnn_25/while/gru_cell_25/BiasAdd_1BiasAdd4backward_rnn_25/while/gru_cell_25/MatMul_1:product:02backward_rnn_25/while/gru_cell_25/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+backward_rnn_25/while/gru_cell_25/BiasAdd_1§
'backward_rnn_25/while/gru_cell_25/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2)
'backward_rnn_25/while/gru_cell_25/Constµ
3backward_rnn_25/while/gru_cell_25/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ25
3backward_rnn_25/while/gru_cell_25/split_1/split_dimþ
)backward_rnn_25/while/gru_cell_25/split_1SplitV4backward_rnn_25/while/gru_cell_25/BiasAdd_1:output:00backward_rnn_25/while/gru_cell_25/Const:output:0<backward_rnn_25/while/gru_cell_25/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2+
)backward_rnn_25/while/gru_cell_25/split_1ð
%backward_rnn_25/while/gru_cell_25/addAddV20backward_rnn_25/while/gru_cell_25/split:output:02backward_rnn_25/while/gru_cell_25/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%backward_rnn_25/while/gru_cell_25/add¿
)backward_rnn_25/while/gru_cell_25/SigmoidSigmoid)backward_rnn_25/while/gru_cell_25/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)backward_rnn_25/while/gru_cell_25/Sigmoidô
'backward_rnn_25/while/gru_cell_25/add_1AddV20backward_rnn_25/while/gru_cell_25/split:output:12backward_rnn_25/while/gru_cell_25/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'backward_rnn_25/while/gru_cell_25/add_1Å
+backward_rnn_25/while/gru_cell_25/Sigmoid_1Sigmoid+backward_rnn_25/while/gru_cell_25/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+backward_rnn_25/while/gru_cell_25/Sigmoid_1í
%backward_rnn_25/while/gru_cell_25/mulMul/backward_rnn_25/while/gru_cell_25/Sigmoid_1:y:02backward_rnn_25/while/gru_cell_25/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%backward_rnn_25/while/gru_cell_25/mulë
'backward_rnn_25/while/gru_cell_25/add_2AddV20backward_rnn_25/while/gru_cell_25/split:output:2)backward_rnn_25/while/gru_cell_25/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'backward_rnn_25/while/gru_cell_25/add_2¸
&backward_rnn_25/while/gru_cell_25/TanhTanh+backward_rnn_25/while/gru_cell_25/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&backward_rnn_25/while/gru_cell_25/Tanhà
'backward_rnn_25/while/gru_cell_25/mul_1Mul-backward_rnn_25/while/gru_cell_25/Sigmoid:y:0#backward_rnn_25_while_placeholder_2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'backward_rnn_25/while/gru_cell_25/mul_1
'backward_rnn_25/while/gru_cell_25/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'backward_rnn_25/while/gru_cell_25/sub/xé
%backward_rnn_25/while/gru_cell_25/subSub0backward_rnn_25/while/gru_cell_25/sub/x:output:0-backward_rnn_25/while/gru_cell_25/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%backward_rnn_25/while/gru_cell_25/subã
'backward_rnn_25/while/gru_cell_25/mul_2Mul)backward_rnn_25/while/gru_cell_25/sub:z:0*backward_rnn_25/while/gru_cell_25/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'backward_rnn_25/while/gru_cell_25/mul_2è
'backward_rnn_25/while/gru_cell_25/add_3AddV2+backward_rnn_25/while/gru_cell_25/mul_1:z:0+backward_rnn_25/while/gru_cell_25/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'backward_rnn_25/while/gru_cell_25/add_3¯
:backward_rnn_25/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#backward_rnn_25_while_placeholder_1!backward_rnn_25_while_placeholder+backward_rnn_25/while/gru_cell_25/add_3:z:0*
_output_shapes
: *
element_dtype02<
:backward_rnn_25/while/TensorArrayV2Write/TensorListSetItem|
backward_rnn_25/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_rnn_25/while/add/y©
backward_rnn_25/while/addAddV2!backward_rnn_25_while_placeholder$backward_rnn_25/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_rnn_25/while/add
backward_rnn_25/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_rnn_25/while/add_1/yÆ
backward_rnn_25/while/add_1AddV28backward_rnn_25_while_backward_rnn_25_while_loop_counter&backward_rnn_25/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_rnn_25/while/add_1·
backward_rnn_25/while/IdentityIdentitybackward_rnn_25/while/add_1:z:08^backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp:^backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp1^backward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2 
backward_rnn_25/while/IdentityÚ
 backward_rnn_25/while/Identity_1Identity>backward_rnn_25_while_backward_rnn_25_while_maximum_iterations8^backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp:^backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp1^backward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2"
 backward_rnn_25/while/Identity_1¹
 backward_rnn_25/while/Identity_2Identitybackward_rnn_25/while/add:z:08^backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp:^backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp1^backward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2"
 backward_rnn_25/while/Identity_2æ
 backward_rnn_25/while/Identity_3IdentityJbackward_rnn_25/while/TensorArrayV2Write/TensorListSetItem:output_handle:08^backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp:^backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp1^backward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2"
 backward_rnn_25/while/Identity_3Ù
 backward_rnn_25/while/Identity_4Identity+backward_rnn_25/while/gru_cell_25/add_3:z:08^backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp:^backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp1^backward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 backward_rnn_25/while/Identity_4"p
5backward_rnn_25_while_backward_rnn_25_strided_slice_17backward_rnn_25_while_backward_rnn_25_strided_slice_1_0"
Bbackward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resourceDbackward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource_0"
@backward_rnn_25_while_gru_cell_25_matmul_readvariableop_resourceBbackward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource_0"x
9backward_rnn_25_while_gru_cell_25_readvariableop_resource;backward_rnn_25_while_gru_cell_25_readvariableop_resource_0"I
backward_rnn_25_while_identity'backward_rnn_25/while/Identity:output:0"M
 backward_rnn_25_while_identity_1)backward_rnn_25/while/Identity_1:output:0"M
 backward_rnn_25_while_identity_2)backward_rnn_25/while/Identity_2:output:0"M
 backward_rnn_25_while_identity_3)backward_rnn_25/while/Identity_3:output:0"M
 backward_rnn_25_while_identity_4)backward_rnn_25/while/Identity_4:output:0"è
qbackward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_backward_rnn_25_tensorarrayunstack_tensorlistfromtensorsbackward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_backward_rnn_25_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : 2r
7backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp7backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp2v
9backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp9backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp2d
0backward_rnn_25/while/gru_cell_25/ReadVariableOp0backward_rnn_25/while/gru_cell_25/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
·
Ð
ITree_Encoder_Double_Bidirectional_GRU_forward_rnn_25_while_body_387378306
tree_encoder_double_bidirectional_gru_forward_rnn_25_while_tree_encoder_double_bidirectional_gru_forward_rnn_25_while_loop_counter
tree_encoder_double_bidirectional_gru_forward_rnn_25_while_tree_encoder_double_bidirectional_gru_forward_rnn_25_while_maximum_iterationsJ
Ftree_encoder_double_bidirectional_gru_forward_rnn_25_while_placeholderL
Htree_encoder_double_bidirectional_gru_forward_rnn_25_while_placeholder_1L
Htree_encoder_double_bidirectional_gru_forward_rnn_25_while_placeholder_2
tree_encoder_double_bidirectional_gru_forward_rnn_25_while_tree_encoder_double_bidirectional_gru_forward_rnn_25_strided_slice_1_0Â
½tree_encoder_double_bidirectional_gru_forward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_tree_encoder_double_bidirectional_gru_forward_rnn_25_tensorarrayunstack_tensorlistfromtensor_0s
`tree_encoder_double_bidirectional_gru_forward_rnn_25_while_gru_cell_25_readvariableop_resource_0:	{
gtree_encoder_double_bidirectional_gru_forward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource_0:
}
itree_encoder_double_bidirectional_gru_forward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource_0:
G
Ctree_encoder_double_bidirectional_gru_forward_rnn_25_while_identityI
Etree_encoder_double_bidirectional_gru_forward_rnn_25_while_identity_1I
Etree_encoder_double_bidirectional_gru_forward_rnn_25_while_identity_2I
Etree_encoder_double_bidirectional_gru_forward_rnn_25_while_identity_3I
Etree_encoder_double_bidirectional_gru_forward_rnn_25_while_identity_4
tree_encoder_double_bidirectional_gru_forward_rnn_25_while_tree_encoder_double_bidirectional_gru_forward_rnn_25_strided_slice_1À
»tree_encoder_double_bidirectional_gru_forward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_tree_encoder_double_bidirectional_gru_forward_rnn_25_tensorarrayunstack_tensorlistfromtensorq
^tree_encoder_double_bidirectional_gru_forward_rnn_25_while_gru_cell_25_readvariableop_resource:	y
etree_encoder_double_bidirectional_gru_forward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource:
{
gtree_encoder_double_bidirectional_gru_forward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource:
¢\Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp¢^Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp¢UTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/ReadVariableOp­
lTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2n
lTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/TensorArrayV2Read/TensorListGetItem/element_shape
^Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem½tree_encoder_double_bidirectional_gru_forward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_tree_encoder_double_bidirectional_gru_forward_rnn_25_tensorarrayunstack_tensorlistfromtensor_0Ftree_encoder_double_bidirectional_gru_forward_rnn_25_while_placeholderuTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02`
^Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/TensorArrayV2Read/TensorListGetItemÐ
UTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/ReadVariableOpReadVariableOp`tree_encoder_double_bidirectional_gru_forward_rnn_25_while_gru_cell_25_readvariableop_resource_0*
_output_shapes
:	*
dtype02W
UTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/ReadVariableOpÁ
NTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/unstackUnpack]Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2P
NTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/unstackæ
\Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOpReadVariableOpgtree_encoder_double_bidirectional_gru_forward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype02^
\Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp¨
MTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMulMatMuleTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/TensorArrayV2Read/TensorListGetItem:item:0dTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2O
MTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul
NTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/BiasAddBiasAddWTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul:product:0WTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2P
NTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/BiasAddû
VTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2X
VTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/split/split_dimË
LTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/splitSplit_Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/split/split_dim:output:0WTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2N
LTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/splitì
^Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpReadVariableOpitree_encoder_double_bidirectional_gru_forward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02`
^Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp
OTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul_1MatMulHtree_encoder_double_bidirectional_gru_forward_rnn_25_while_placeholder_2fTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Q
OTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul_1
PTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/BiasAdd_1BiasAddYTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul_1:product:0WTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2R
PTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/BiasAdd_1ñ
LTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2N
LTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/Constÿ
XTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2Z
XTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/split_1/split_dim·
NTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/split_1SplitVYTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/BiasAdd_1:output:0UTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/Const:output:0aTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2P
NTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/split_1
JTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/addAddV2UTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/split:output:0WTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2L
JTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/add®
NTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/SigmoidSigmoidNTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2P
NTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/Sigmoid
LTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/add_1AddV2UTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/split:output:1WTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2N
LTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/add_1´
PTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/Sigmoid_1SigmoidPTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2R
PTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/Sigmoid_1
JTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/mulMulTTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/Sigmoid_1:y:0WTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2L
JTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/mulÿ
LTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/add_2AddV2UTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/split:output:2NTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2N
LTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/add_2§
KTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/TanhTanhPTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2M
KTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/Tanhô
LTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/mul_1MulRTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/Sigmoid:y:0Htree_encoder_double_bidirectional_gru_forward_rnn_25_while_placeholder_2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2N
LTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/mul_1á
LTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2N
LTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/sub/xý
JTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/subSubUTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/sub/x:output:0RTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2L
JTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/sub÷
LTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/mul_2MulNTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/sub:z:0OTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2N
LTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/mul_2ü
LTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/add_3AddV2PTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/mul_1:z:0PTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2N
LTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/add_3è
_Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemHtree_encoder_double_bidirectional_gru_forward_rnn_25_while_placeholder_1Ftree_encoder_double_bidirectional_gru_forward_rnn_25_while_placeholderPTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/add_3:z:0*
_output_shapes
: *
element_dtype02a
_Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/TensorArrayV2Write/TensorListSetItemÆ
@Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2B
@Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/add/y½
>Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/addAddV2Ftree_encoder_double_bidirectional_gru_forward_rnn_25_while_placeholderITree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/add/y:output:0*
T0*
_output_shapes
: 2@
>Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/addÊ
BTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2D
BTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/add_1/y
@Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/add_1AddV2tree_encoder_double_bidirectional_gru_forward_rnn_25_while_tree_encoder_double_bidirectional_gru_forward_rnn_25_while_loop_counterKTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/add_1/y:output:0*
T0*
_output_shapes
: 2B
@Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/add_1
CTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/IdentityIdentityDTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/add_1:z:0]^Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp_^Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpV^Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2E
CTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/IdentityÞ
ETree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/Identity_1Identitytree_encoder_double_bidirectional_gru_forward_rnn_25_while_tree_encoder_double_bidirectional_gru_forward_rnn_25_while_maximum_iterations]^Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp_^Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpV^Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2G
ETree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/Identity_1
ETree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/Identity_2IdentityBTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/add:z:0]^Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp_^Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpV^Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2G
ETree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/Identity_2Ä
ETree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/Identity_3IdentityoTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/TensorArrayV2Write/TensorListSetItem:output_handle:0]^Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp_^Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpV^Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2G
ETree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/Identity_3·
ETree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/Identity_4IdentityPTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/add_3:z:0]^Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp_^Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpV^Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2G
ETree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/Identity_4"Ô
gtree_encoder_double_bidirectional_gru_forward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resourceitree_encoder_double_bidirectional_gru_forward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource_0"Ð
etree_encoder_double_bidirectional_gru_forward_rnn_25_while_gru_cell_25_matmul_readvariableop_resourcegtree_encoder_double_bidirectional_gru_forward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource_0"Â
^tree_encoder_double_bidirectional_gru_forward_rnn_25_while_gru_cell_25_readvariableop_resource`tree_encoder_double_bidirectional_gru_forward_rnn_25_while_gru_cell_25_readvariableop_resource_0"
Ctree_encoder_double_bidirectional_gru_forward_rnn_25_while_identityLTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/Identity:output:0"
Etree_encoder_double_bidirectional_gru_forward_rnn_25_while_identity_1NTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/Identity_1:output:0"
Etree_encoder_double_bidirectional_gru_forward_rnn_25_while_identity_2NTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/Identity_2:output:0"
Etree_encoder_double_bidirectional_gru_forward_rnn_25_while_identity_3NTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/Identity_3:output:0"
Etree_encoder_double_bidirectional_gru_forward_rnn_25_while_identity_4NTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/Identity_4:output:0"þ
»tree_encoder_double_bidirectional_gru_forward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_tree_encoder_double_bidirectional_gru_forward_rnn_25_tensorarrayunstack_tensorlistfromtensor½tree_encoder_double_bidirectional_gru_forward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_tree_encoder_double_bidirectional_gru_forward_rnn_25_tensorarrayunstack_tensorlistfromtensor_0"
tree_encoder_double_bidirectional_gru_forward_rnn_25_while_tree_encoder_double_bidirectional_gru_forward_rnn_25_strided_slice_1tree_encoder_double_bidirectional_gru_forward_rnn_25_while_tree_encoder_double_bidirectional_gru_forward_rnn_25_strided_slice_1_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : 2¼
\Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp\Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp2À
^Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp^Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp2®
UTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/ReadVariableOpUTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
öZ

M__inference_forward_rnn_25_layer_call_and_return_conditional_losses_387382361
inputs_06
#gru_cell_25_readvariableop_resource:	>
*gru_cell_25_matmul_readvariableop_resource:
@
,gru_cell_25_matmul_1_readvariableop_resource:

identity¢!gru_cell_25/MatMul/ReadVariableOp¢#gru_cell_25/MatMul_1/ReadVariableOp¢gru_cell_25/ReadVariableOp¢whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2
gru_cell_25/ReadVariableOpReadVariableOp#gru_cell_25_readvariableop_resource*
_output_shapes
:	*
dtype02
gru_cell_25/ReadVariableOp
gru_cell_25/unstackUnpack"gru_cell_25/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
gru_cell_25/unstack³
!gru_cell_25/MatMul/ReadVariableOpReadVariableOp*gru_cell_25_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02#
!gru_cell_25/MatMul/ReadVariableOpª
gru_cell_25/MatMulMatMulstrided_slice_2:output:0)gru_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/MatMul¤
gru_cell_25/BiasAddBiasAddgru_cell_25/MatMul:product:0gru_cell_25/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/BiasAdd
gru_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/split/split_dimß
gru_cell_25/splitSplit$gru_cell_25/split/split_dim:output:0gru_cell_25/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
gru_cell_25/split¹
#gru_cell_25/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_25_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02%
#gru_cell_25/MatMul_1/ReadVariableOp¦
gru_cell_25/MatMul_1MatMulzeros:output:0+gru_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/MatMul_1ª
gru_cell_25/BiasAdd_1BiasAddgru_cell_25/MatMul_1:product:0gru_cell_25/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/BiasAdd_1{
gru_cell_25/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2
gru_cell_25/Const
gru_cell_25/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/split_1/split_dim
gru_cell_25/split_1SplitVgru_cell_25/BiasAdd_1:output:0gru_cell_25/Const:output:0&gru_cell_25/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
gru_cell_25/split_1
gru_cell_25/addAddV2gru_cell_25/split:output:0gru_cell_25/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/add}
gru_cell_25/SigmoidSigmoidgru_cell_25/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/Sigmoid
gru_cell_25/add_1AddV2gru_cell_25/split:output:1gru_cell_25/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/add_1
gru_cell_25/Sigmoid_1Sigmoidgru_cell_25/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/Sigmoid_1
gru_cell_25/mulMulgru_cell_25/Sigmoid_1:y:0gru_cell_25/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/mul
gru_cell_25/add_2AddV2gru_cell_25/split:output:2gru_cell_25/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/add_2v
gru_cell_25/TanhTanhgru_cell_25/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/Tanh
gru_cell_25/mul_1Mulgru_cell_25/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/mul_1k
gru_cell_25/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_25/sub/x
gru_cell_25/subSubgru_cell_25/sub/x:output:0gru_cell_25/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/sub
gru_cell_25/mul_2Mulgru_cell_25/sub:z:0gru_cell_25/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/mul_2
gru_cell_25/add_3AddV2gru_cell_25/mul_1:z:0gru_cell_25/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter²
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_25_readvariableop_resource*gru_cell_25_matmul_readvariableop_resource,gru_cell_25_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	* 
bodyR
while_body_387382273* 
condR
while_cond_387382272*9
output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
transpose_1Ü
IdentityIdentitystrided_slice_3:output:0"^gru_cell_25/MatMul/ReadVariableOp$^gru_cell_25/MatMul_1/ReadVariableOp^gru_cell_25/ReadVariableOp^while*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2F
!gru_cell_25/MatMul/ReadVariableOp!gru_cell_25/MatMul/ReadVariableOp2J
#gru_cell_25/MatMul_1/ReadVariableOp#gru_cell_25/MatMul_1/ReadVariableOp28
gru_cell_25/ReadVariableOpgru_cell_25/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0

È
3__inference_backward_rnn_25_layer_call_fn_387382383
inputs_0
unknown:	
	unknown_0:

	unknown_1:

identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_backward_rnn_25_layer_call_and_return_conditional_losses_3873794852
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0

ï
JTree_Encoder_Double_Bidirectional_GRU_backward_rnn_25_while_body_387378460
tree_encoder_double_bidirectional_gru_backward_rnn_25_while_tree_encoder_double_bidirectional_gru_backward_rnn_25_while_loop_counter
tree_encoder_double_bidirectional_gru_backward_rnn_25_while_tree_encoder_double_bidirectional_gru_backward_rnn_25_while_maximum_iterationsK
Gtree_encoder_double_bidirectional_gru_backward_rnn_25_while_placeholderM
Itree_encoder_double_bidirectional_gru_backward_rnn_25_while_placeholder_1M
Itree_encoder_double_bidirectional_gru_backward_rnn_25_while_placeholder_2
tree_encoder_double_bidirectional_gru_backward_rnn_25_while_tree_encoder_double_bidirectional_gru_backward_rnn_25_strided_slice_1_0Ä
¿tree_encoder_double_bidirectional_gru_backward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_tree_encoder_double_bidirectional_gru_backward_rnn_25_tensorarrayunstack_tensorlistfromtensor_0t
atree_encoder_double_bidirectional_gru_backward_rnn_25_while_gru_cell_25_readvariableop_resource_0:	|
htree_encoder_double_bidirectional_gru_backward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource_0:
~
jtree_encoder_double_bidirectional_gru_backward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource_0:
H
Dtree_encoder_double_bidirectional_gru_backward_rnn_25_while_identityJ
Ftree_encoder_double_bidirectional_gru_backward_rnn_25_while_identity_1J
Ftree_encoder_double_bidirectional_gru_backward_rnn_25_while_identity_2J
Ftree_encoder_double_bidirectional_gru_backward_rnn_25_while_identity_3J
Ftree_encoder_double_bidirectional_gru_backward_rnn_25_while_identity_4
tree_encoder_double_bidirectional_gru_backward_rnn_25_while_tree_encoder_double_bidirectional_gru_backward_rnn_25_strided_slice_1Â
½tree_encoder_double_bidirectional_gru_backward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_tree_encoder_double_bidirectional_gru_backward_rnn_25_tensorarrayunstack_tensorlistfromtensorr
_tree_encoder_double_bidirectional_gru_backward_rnn_25_while_gru_cell_25_readvariableop_resource:	z
ftree_encoder_double_bidirectional_gru_backward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource:
|
htree_encoder_double_bidirectional_gru_backward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource:
¢]Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp¢_Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp¢VTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/ReadVariableOp¯
mTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2o
mTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/TensorArrayV2Read/TensorListGetItem/element_shape
_Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem¿tree_encoder_double_bidirectional_gru_backward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_tree_encoder_double_bidirectional_gru_backward_rnn_25_tensorarrayunstack_tensorlistfromtensor_0Gtree_encoder_double_bidirectional_gru_backward_rnn_25_while_placeholdervTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02a
_Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/TensorArrayV2Read/TensorListGetItemÓ
VTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/ReadVariableOpReadVariableOpatree_encoder_double_bidirectional_gru_backward_rnn_25_while_gru_cell_25_readvariableop_resource_0*
_output_shapes
:	*
dtype02X
VTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/ReadVariableOpÄ
OTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/unstackUnpack^Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2Q
OTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/unstacké
]Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOpReadVariableOphtree_encoder_double_bidirectional_gru_backward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype02_
]Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp¬
NTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMulMatMulfTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/TensorArrayV2Read/TensorListGetItem:item:0eTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2P
NTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul
OTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/BiasAddBiasAddXTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul:product:0XTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Q
OTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/BiasAddý
WTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2Y
WTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/split/split_dimÏ
MTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/splitSplit`Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/split/split_dim:output:0XTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2O
MTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/splitï
_Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpReadVariableOpjtree_encoder_double_bidirectional_gru_backward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02a
_Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp
PTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul_1MatMulItree_encoder_double_bidirectional_gru_backward_rnn_25_while_placeholder_2gTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2R
PTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul_1
QTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/BiasAdd_1BiasAddZTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul_1:product:0XTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2S
QTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/BiasAdd_1ó
MTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2O
MTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/Const
YTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2[
YTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/split_1/split_dim¼
OTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/split_1SplitVZTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/BiasAdd_1:output:0VTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/Const:output:0bTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2Q
OTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/split_1
KTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/addAddV2VTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/split:output:0XTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2M
KTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/add±
OTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/SigmoidSigmoidOTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Q
OTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/Sigmoid
MTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/add_1AddV2VTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/split:output:1XTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2O
MTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/add_1·
QTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/Sigmoid_1SigmoidQTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2S
QTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/Sigmoid_1
KTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/mulMulUTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/Sigmoid_1:y:0XTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2M
KTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/mul
MTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/add_2AddV2VTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/split:output:2OTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2O
MTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/add_2ª
LTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/TanhTanhQTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2N
LTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/Tanhø
MTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/mul_1MulSTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/Sigmoid:y:0Itree_encoder_double_bidirectional_gru_backward_rnn_25_while_placeholder_2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2O
MTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/mul_1ã
MTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2O
MTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/sub/x
KTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/subSubVTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/sub/x:output:0STree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2M
KTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/subû
MTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/mul_2MulOTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/sub:z:0PTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2O
MTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/mul_2
MTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/add_3AddV2QTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/mul_1:z:0QTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2O
MTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/add_3í
`Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemItree_encoder_double_bidirectional_gru_backward_rnn_25_while_placeholder_1Gtree_encoder_double_bidirectional_gru_backward_rnn_25_while_placeholderQTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/add_3:z:0*
_output_shapes
: *
element_dtype02b
`Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/TensorArrayV2Write/TensorListSetItemÈ
ATree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2C
ATree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/add/yÁ
?Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/addAddV2Gtree_encoder_double_bidirectional_gru_backward_rnn_25_while_placeholderJTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/add/y:output:0*
T0*
_output_shapes
: 2A
?Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/addÌ
CTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2E
CTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/add_1/y
ATree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/add_1AddV2tree_encoder_double_bidirectional_gru_backward_rnn_25_while_tree_encoder_double_bidirectional_gru_backward_rnn_25_while_loop_counterLTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/add_1/y:output:0*
T0*
_output_shapes
: 2C
ATree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/add_1
DTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/IdentityIdentityETree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/add_1:z:0^^Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp`^Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpW^Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2F
DTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/Identityå
FTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/Identity_1Identitytree_encoder_double_bidirectional_gru_backward_rnn_25_while_tree_encoder_double_bidirectional_gru_backward_rnn_25_while_maximum_iterations^^Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp`^Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpW^Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2H
FTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/Identity_1
FTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/Identity_2IdentityCTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/add:z:0^^Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp`^Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpW^Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2H
FTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/Identity_2Ê
FTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/Identity_3IdentitypTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^^Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp`^Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpW^Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2H
FTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/Identity_3½
FTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/Identity_4IdentityQTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/add_3:z:0^^Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp`^Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpW^Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2H
FTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/Identity_4"Ö
htree_encoder_double_bidirectional_gru_backward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resourcejtree_encoder_double_bidirectional_gru_backward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource_0"Ò
ftree_encoder_double_bidirectional_gru_backward_rnn_25_while_gru_cell_25_matmul_readvariableop_resourcehtree_encoder_double_bidirectional_gru_backward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource_0"Ä
_tree_encoder_double_bidirectional_gru_backward_rnn_25_while_gru_cell_25_readvariableop_resourceatree_encoder_double_bidirectional_gru_backward_rnn_25_while_gru_cell_25_readvariableop_resource_0"
Dtree_encoder_double_bidirectional_gru_backward_rnn_25_while_identityMTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/Identity:output:0"
Ftree_encoder_double_bidirectional_gru_backward_rnn_25_while_identity_1OTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/Identity_1:output:0"
Ftree_encoder_double_bidirectional_gru_backward_rnn_25_while_identity_2OTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/Identity_2:output:0"
Ftree_encoder_double_bidirectional_gru_backward_rnn_25_while_identity_3OTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/Identity_3:output:0"
Ftree_encoder_double_bidirectional_gru_backward_rnn_25_while_identity_4OTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/Identity_4:output:0"
½tree_encoder_double_bidirectional_gru_backward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_tree_encoder_double_bidirectional_gru_backward_rnn_25_tensorarrayunstack_tensorlistfromtensor¿tree_encoder_double_bidirectional_gru_backward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_tree_encoder_double_bidirectional_gru_backward_rnn_25_tensorarrayunstack_tensorlistfromtensor_0"
tree_encoder_double_bidirectional_gru_backward_rnn_25_while_tree_encoder_double_bidirectional_gru_backward_rnn_25_strided_slice_1tree_encoder_double_bidirectional_gru_backward_rnn_25_while_tree_encoder_double_bidirectional_gru_backward_rnn_25_strided_slice_1_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : 2¾
]Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp]Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp2Â
_Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp_Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp2°
VTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/ReadVariableOpVTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 

È
3__inference_backward_rnn_25_layer_call_fn_387382372
inputs_0
unknown:	
	unknown_0:

	unknown_1:

identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_backward_rnn_25_layer_call_and_return_conditional_losses_3873792872
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0

¹
while_cond_387379223
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_17
3while_while_cond_387379223___redundant_placeholder07
3while_while_cond_387379223___redundant_placeholder17
3while_while_cond_387379223___redundant_placeholder27
3while_while_cond_387379223___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ß
à
J__inference_gru_cell_25_layer_call_and_return_conditional_losses_387382766

inputs
states_0*
readvariableop_resource:	2
matmul_readvariableop_resource:
4
 matmul_1_readvariableop_resource:

identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpy
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
split/split_dim¯
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	BiasAdd_1c
ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2
Constq
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
split_1/split_dimÔ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanh_
mul_1MulSigmoid:y:0states_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sub[
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_3¡
IdentityIdentity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¥

Identity_1Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0
 	
«
<__inference_Double_Bidirectional_GRU_layer_call_fn_387381383

inputs
unknown:	
	unknown_0:

	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:

identity¢StatefulPartitionedCall¿
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *`
f[RY
W__inference_Double_Bidirectional_GRU_layer_call_and_return_conditional_losses_3873804402
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ß
à
J__inference_gru_cell_25_layer_call_and_return_conditional_losses_387382872

inputs
states_0*
readvariableop_resource:	2
matmul_readvariableop_resource:
4
 matmul_1_readvariableop_resource:

identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpy
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
split/split_dim¯
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	BiasAdd_1c
ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2
Constq
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
split_1/split_dimÔ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanh_
mul_1MulSigmoid:y:0states_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sub[
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_3¡
IdentityIdentity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¥

Identity_1Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0
Ý
é
$backward_rnn_25_while_cond_387379956<
8backward_rnn_25_while_backward_rnn_25_while_loop_counterB
>backward_rnn_25_while_backward_rnn_25_while_maximum_iterations%
!backward_rnn_25_while_placeholder'
#backward_rnn_25_while_placeholder_1'
#backward_rnn_25_while_placeholder_2>
:backward_rnn_25_while_less_backward_rnn_25_strided_slice_1W
Sbackward_rnn_25_while_backward_rnn_25_while_cond_387379956___redundant_placeholder0W
Sbackward_rnn_25_while_backward_rnn_25_while_cond_387379956___redundant_placeholder1W
Sbackward_rnn_25_while_backward_rnn_25_while_cond_387379956___redundant_placeholder2W
Sbackward_rnn_25_while_backward_rnn_25_while_cond_387379956___redundant_placeholder3"
backward_rnn_25_while_identity
À
backward_rnn_25/while/LessLess!backward_rnn_25_while_placeholder:backward_rnn_25_while_less_backward_rnn_25_strided_slice_1*
T0*
_output_shapes
: 2
backward_rnn_25/while/Less
backward_rnn_25/while/IdentityIdentitybackward_rnn_25/while/Less:z:0*
T0
*
_output_shapes
: 2 
backward_rnn_25/while/Identity"I
backward_rnn_25_while_identity'backward_rnn_25/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Ù
Ã
<Double_Bidirectional_GRU_forward_rnn_25_while_body_387381071l
hdouble_bidirectional_gru_forward_rnn_25_while_double_bidirectional_gru_forward_rnn_25_while_loop_counterr
ndouble_bidirectional_gru_forward_rnn_25_while_double_bidirectional_gru_forward_rnn_25_while_maximum_iterations=
9double_bidirectional_gru_forward_rnn_25_while_placeholder?
;double_bidirectional_gru_forward_rnn_25_while_placeholder_1?
;double_bidirectional_gru_forward_rnn_25_while_placeholder_2k
gdouble_bidirectional_gru_forward_rnn_25_while_double_bidirectional_gru_forward_rnn_25_strided_slice_1_0¨
£double_bidirectional_gru_forward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_double_bidirectional_gru_forward_rnn_25_tensorarrayunstack_tensorlistfromtensor_0f
Sdouble_bidirectional_gru_forward_rnn_25_while_gru_cell_25_readvariableop_resource_0:	n
Zdouble_bidirectional_gru_forward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource_0:
p
\double_bidirectional_gru_forward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource_0:
:
6double_bidirectional_gru_forward_rnn_25_while_identity<
8double_bidirectional_gru_forward_rnn_25_while_identity_1<
8double_bidirectional_gru_forward_rnn_25_while_identity_2<
8double_bidirectional_gru_forward_rnn_25_while_identity_3<
8double_bidirectional_gru_forward_rnn_25_while_identity_4i
edouble_bidirectional_gru_forward_rnn_25_while_double_bidirectional_gru_forward_rnn_25_strided_slice_1¦
¡double_bidirectional_gru_forward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_double_bidirectional_gru_forward_rnn_25_tensorarrayunstack_tensorlistfromtensord
Qdouble_bidirectional_gru_forward_rnn_25_while_gru_cell_25_readvariableop_resource:	l
Xdouble_bidirectional_gru_forward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource:
n
Zdouble_bidirectional_gru_forward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource:
¢ODouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp¢QDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp¢HDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/ReadVariableOp
_Double_Bidirectional_GRU/forward_rnn_25/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2a
_Double_Bidirectional_GRU/forward_rnn_25/while/TensorArrayV2Read/TensorListGetItem/element_shapeÅ
QDouble_Bidirectional_GRU/forward_rnn_25/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem£double_bidirectional_gru_forward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_double_bidirectional_gru_forward_rnn_25_tensorarrayunstack_tensorlistfromtensor_09double_bidirectional_gru_forward_rnn_25_while_placeholderhDouble_Bidirectional_GRU/forward_rnn_25/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02S
QDouble_Bidirectional_GRU/forward_rnn_25/while/TensorArrayV2Read/TensorListGetItem©
HDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/ReadVariableOpReadVariableOpSdouble_bidirectional_gru_forward_rnn_25_while_gru_cell_25_readvariableop_resource_0*
_output_shapes
:	*
dtype02J
HDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/ReadVariableOp
ADouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/unstackUnpackPDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2C
ADouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/unstack¿
ODouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOpReadVariableOpZdouble_bidirectional_gru_forward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype02Q
ODouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOpô
@Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMulMatMulXDouble_Bidirectional_GRU/forward_rnn_25/while/TensorArrayV2Read/TensorListGetItem:item:0WDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2B
@Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMulÜ
ADouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/BiasAddBiasAddJDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul:product:0JDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2C
ADouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/BiasAddá
IDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2K
IDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/split/split_dim
?Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/splitSplitRDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/split/split_dim:output:0JDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2A
?Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/splitÅ
QDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpReadVariableOp\double_bidirectional_gru_forward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02S
QDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpÝ
BDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul_1MatMul;double_bidirectional_gru_forward_rnn_25_while_placeholder_2YDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2D
BDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul_1â
CDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/BiasAdd_1BiasAddLDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul_1:product:0JDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2E
CDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/BiasAdd_1×
?Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2A
?Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/Constå
KDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2M
KDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/split_1/split_dimö
ADouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/split_1SplitVLDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/BiasAdd_1:output:0HDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/Const:output:0TDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2C
ADouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/split_1Ð
=Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/addAddV2HDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/split:output:0JDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2?
=Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/add
ADouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/SigmoidSigmoidADouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2C
ADouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/SigmoidÔ
?Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/add_1AddV2HDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/split:output:1JDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2A
?Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/add_1
CDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/Sigmoid_1SigmoidCDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2E
CDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/Sigmoid_1Í
=Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/mulMulGDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/Sigmoid_1:y:0JDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2?
=Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/mulË
?Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/add_2AddV2HDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/split:output:2ADouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2A
?Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/add_2
>Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/TanhTanhCDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2@
>Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/TanhÀ
?Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/mul_1MulEDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/Sigmoid:y:0;double_bidirectional_gru_forward_rnn_25_while_placeholder_2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2A
?Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/mul_1Ç
?Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2A
?Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/sub/xÉ
=Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/subSubHDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/sub/x:output:0EDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2?
=Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/subÃ
?Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/mul_2MulADouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/sub:z:0BDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2A
?Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/mul_2È
?Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/add_3AddV2CDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/mul_1:z:0CDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2A
?Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/add_3§
RDouble_Bidirectional_GRU/forward_rnn_25/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem;double_bidirectional_gru_forward_rnn_25_while_placeholder_19double_bidirectional_gru_forward_rnn_25_while_placeholderCDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/add_3:z:0*
_output_shapes
: *
element_dtype02T
RDouble_Bidirectional_GRU/forward_rnn_25/while/TensorArrayV2Write/TensorListSetItem¬
3Double_Bidirectional_GRU/forward_rnn_25/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :25
3Double_Bidirectional_GRU/forward_rnn_25/while/add/y
1Double_Bidirectional_GRU/forward_rnn_25/while/addAddV29double_bidirectional_gru_forward_rnn_25_while_placeholder<Double_Bidirectional_GRU/forward_rnn_25/while/add/y:output:0*
T0*
_output_shapes
: 23
1Double_Bidirectional_GRU/forward_rnn_25/while/add°
5Double_Bidirectional_GRU/forward_rnn_25/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :27
5Double_Bidirectional_GRU/forward_rnn_25/while/add_1/y¾
3Double_Bidirectional_GRU/forward_rnn_25/while/add_1AddV2hdouble_bidirectional_gru_forward_rnn_25_while_double_bidirectional_gru_forward_rnn_25_while_loop_counter>Double_Bidirectional_GRU/forward_rnn_25/while/add_1/y:output:0*
T0*
_output_shapes
: 25
3Double_Bidirectional_GRU/forward_rnn_25/while/add_1Ç
6Double_Bidirectional_GRU/forward_rnn_25/while/IdentityIdentity7Double_Bidirectional_GRU/forward_rnn_25/while/add_1:z:0P^Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOpR^Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpI^Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 28
6Double_Bidirectional_GRU/forward_rnn_25/while/Identity
8Double_Bidirectional_GRU/forward_rnn_25/while/Identity_1Identityndouble_bidirectional_gru_forward_rnn_25_while_double_bidirectional_gru_forward_rnn_25_while_maximum_iterationsP^Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOpR^Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpI^Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2:
8Double_Bidirectional_GRU/forward_rnn_25/while/Identity_1É
8Double_Bidirectional_GRU/forward_rnn_25/while/Identity_2Identity5Double_Bidirectional_GRU/forward_rnn_25/while/add:z:0P^Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOpR^Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpI^Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2:
8Double_Bidirectional_GRU/forward_rnn_25/while/Identity_2ö
8Double_Bidirectional_GRU/forward_rnn_25/while/Identity_3IdentitybDouble_Bidirectional_GRU/forward_rnn_25/while/TensorArrayV2Write/TensorListSetItem:output_handle:0P^Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOpR^Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpI^Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2:
8Double_Bidirectional_GRU/forward_rnn_25/while/Identity_3é
8Double_Bidirectional_GRU/forward_rnn_25/while/Identity_4IdentityCDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/add_3:z:0P^Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOpR^Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpI^Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:
8Double_Bidirectional_GRU/forward_rnn_25/while/Identity_4"Ð
edouble_bidirectional_gru_forward_rnn_25_while_double_bidirectional_gru_forward_rnn_25_strided_slice_1gdouble_bidirectional_gru_forward_rnn_25_while_double_bidirectional_gru_forward_rnn_25_strided_slice_1_0"º
Zdouble_bidirectional_gru_forward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource\double_bidirectional_gru_forward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource_0"¶
Xdouble_bidirectional_gru_forward_rnn_25_while_gru_cell_25_matmul_readvariableop_resourceZdouble_bidirectional_gru_forward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource_0"¨
Qdouble_bidirectional_gru_forward_rnn_25_while_gru_cell_25_readvariableop_resourceSdouble_bidirectional_gru_forward_rnn_25_while_gru_cell_25_readvariableop_resource_0"y
6double_bidirectional_gru_forward_rnn_25_while_identity?Double_Bidirectional_GRU/forward_rnn_25/while/Identity:output:0"}
8double_bidirectional_gru_forward_rnn_25_while_identity_1ADouble_Bidirectional_GRU/forward_rnn_25/while/Identity_1:output:0"}
8double_bidirectional_gru_forward_rnn_25_while_identity_2ADouble_Bidirectional_GRU/forward_rnn_25/while/Identity_2:output:0"}
8double_bidirectional_gru_forward_rnn_25_while_identity_3ADouble_Bidirectional_GRU/forward_rnn_25/while/Identity_3:output:0"}
8double_bidirectional_gru_forward_rnn_25_while_identity_4ADouble_Bidirectional_GRU/forward_rnn_25/while/Identity_4:output:0"Ê
¡double_bidirectional_gru_forward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_double_bidirectional_gru_forward_rnn_25_tensorarrayunstack_tensorlistfromtensor£double_bidirectional_gru_forward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_double_bidirectional_gru_forward_rnn_25_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : 2¢
ODouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOpODouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp2¦
QDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpQDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp2
HDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/ReadVariableOpHDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
 	
«
<__inference_Double_Bidirectional_GRU_layer_call_fn_387381366

inputs
unknown:	
	unknown_0:

	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:

identity¢StatefulPartitionedCall¿
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *`
f[RY
W__inference_Double_Bidirectional_GRU_layer_call_and_return_conditional_losses_3873800472
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ç
²
K__inference_Tree_Encoder_layer_call_and_return_conditional_losses_387381322

inputs^
Kdouble_bidirectional_gru_forward_rnn_25_gru_cell_25_readvariableop_resource:	f
Rdouble_bidirectional_gru_forward_rnn_25_gru_cell_25_matmul_readvariableop_resource:
h
Tdouble_bidirectional_gru_forward_rnn_25_gru_cell_25_matmul_1_readvariableop_resource:
_
Ldouble_bidirectional_gru_backward_rnn_25_gru_cell_25_readvariableop_resource:	g
Sdouble_bidirectional_gru_backward_rnn_25_gru_cell_25_matmul_readvariableop_resource:
i
Udouble_bidirectional_gru_backward_rnn_25_gru_cell_25_matmul_1_readvariableop_resource:
<
)prediction_matmul_readvariableop_resource:	8
*prediction_biasadd_readvariableop_resource:
identity¢JDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMul/ReadVariableOp¢LDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp¢CDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/ReadVariableOp¢.Double_Bidirectional_GRU/backward_rnn_25/while¢IDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMul/ReadVariableOp¢KDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp¢BDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/ReadVariableOp¢-Double_Bidirectional_GRU/forward_rnn_25/while¢!Prediction/BiasAdd/ReadVariableOp¢ Prediction/MatMul/ReadVariableOp
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
Embedding_Dropout/dropout/Mul_1±
-Double_Bidirectional_GRU/forward_rnn_25/ShapeShape#Embedding_Dropout/dropout/Mul_1:z:0*
T0*
_output_shapes
:2/
-Double_Bidirectional_GRU/forward_rnn_25/ShapeÄ
;Double_Bidirectional_GRU/forward_rnn_25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2=
;Double_Bidirectional_GRU/forward_rnn_25/strided_slice/stackÈ
=Double_Bidirectional_GRU/forward_rnn_25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2?
=Double_Bidirectional_GRU/forward_rnn_25/strided_slice/stack_1È
=Double_Bidirectional_GRU/forward_rnn_25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=Double_Bidirectional_GRU/forward_rnn_25/strided_slice/stack_2Ò
5Double_Bidirectional_GRU/forward_rnn_25/strided_sliceStridedSlice6Double_Bidirectional_GRU/forward_rnn_25/Shape:output:0DDouble_Bidirectional_GRU/forward_rnn_25/strided_slice/stack:output:0FDouble_Bidirectional_GRU/forward_rnn_25/strided_slice/stack_1:output:0FDouble_Bidirectional_GRU/forward_rnn_25/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask27
5Double_Bidirectional_GRU/forward_rnn_25/strided_slice­
3Double_Bidirectional_GRU/forward_rnn_25/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :25
3Double_Bidirectional_GRU/forward_rnn_25/zeros/mul/y
1Double_Bidirectional_GRU/forward_rnn_25/zeros/mulMul>Double_Bidirectional_GRU/forward_rnn_25/strided_slice:output:0<Double_Bidirectional_GRU/forward_rnn_25/zeros/mul/y:output:0*
T0*
_output_shapes
: 23
1Double_Bidirectional_GRU/forward_rnn_25/zeros/mul¯
4Double_Bidirectional_GRU/forward_rnn_25/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è26
4Double_Bidirectional_GRU/forward_rnn_25/zeros/Less/y
2Double_Bidirectional_GRU/forward_rnn_25/zeros/LessLess5Double_Bidirectional_GRU/forward_rnn_25/zeros/mul:z:0=Double_Bidirectional_GRU/forward_rnn_25/zeros/Less/y:output:0*
T0*
_output_shapes
: 24
2Double_Bidirectional_GRU/forward_rnn_25/zeros/Less³
6Double_Bidirectional_GRU/forward_rnn_25/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :28
6Double_Bidirectional_GRU/forward_rnn_25/zeros/packed/1£
4Double_Bidirectional_GRU/forward_rnn_25/zeros/packedPack>Double_Bidirectional_GRU/forward_rnn_25/strided_slice:output:0?Double_Bidirectional_GRU/forward_rnn_25/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:26
4Double_Bidirectional_GRU/forward_rnn_25/zeros/packed¯
3Double_Bidirectional_GRU/forward_rnn_25/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    25
3Double_Bidirectional_GRU/forward_rnn_25/zeros/Const
-Double_Bidirectional_GRU/forward_rnn_25/zerosFill=Double_Bidirectional_GRU/forward_rnn_25/zeros/packed:output:0<Double_Bidirectional_GRU/forward_rnn_25/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-Double_Bidirectional_GRU/forward_rnn_25/zerosÅ
6Double_Bidirectional_GRU/forward_rnn_25/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          28
6Double_Bidirectional_GRU/forward_rnn_25/transpose/perm
1Double_Bidirectional_GRU/forward_rnn_25/transpose	Transpose#Embedding_Dropout/dropout/Mul_1:z:0?Double_Bidirectional_GRU/forward_rnn_25/transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ23
1Double_Bidirectional_GRU/forward_rnn_25/transposeÇ
/Double_Bidirectional_GRU/forward_rnn_25/Shape_1Shape5Double_Bidirectional_GRU/forward_rnn_25/transpose:y:0*
T0*
_output_shapes
:21
/Double_Bidirectional_GRU/forward_rnn_25/Shape_1È
=Double_Bidirectional_GRU/forward_rnn_25/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=Double_Bidirectional_GRU/forward_rnn_25/strided_slice_1/stackÌ
?Double_Bidirectional_GRU/forward_rnn_25/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2A
?Double_Bidirectional_GRU/forward_rnn_25/strided_slice_1/stack_1Ì
?Double_Bidirectional_GRU/forward_rnn_25/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?Double_Bidirectional_GRU/forward_rnn_25/strided_slice_1/stack_2Þ
7Double_Bidirectional_GRU/forward_rnn_25/strided_slice_1StridedSlice8Double_Bidirectional_GRU/forward_rnn_25/Shape_1:output:0FDouble_Bidirectional_GRU/forward_rnn_25/strided_slice_1/stack:output:0HDouble_Bidirectional_GRU/forward_rnn_25/strided_slice_1/stack_1:output:0HDouble_Bidirectional_GRU/forward_rnn_25/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask29
7Double_Bidirectional_GRU/forward_rnn_25/strided_slice_1Õ
CDouble_Bidirectional_GRU/forward_rnn_25/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2E
CDouble_Bidirectional_GRU/forward_rnn_25/TensorArrayV2/element_shapeÒ
5Double_Bidirectional_GRU/forward_rnn_25/TensorArrayV2TensorListReserveLDouble_Bidirectional_GRU/forward_rnn_25/TensorArrayV2/element_shape:output:0@Double_Bidirectional_GRU/forward_rnn_25/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type027
5Double_Bidirectional_GRU/forward_rnn_25/TensorArrayV2
]Double_Bidirectional_GRU/forward_rnn_25/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2_
]Double_Bidirectional_GRU/forward_rnn_25/TensorArrayUnstack/TensorListFromTensor/element_shape
ODouble_Bidirectional_GRU/forward_rnn_25/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor5Double_Bidirectional_GRU/forward_rnn_25/transpose:y:0fDouble_Bidirectional_GRU/forward_rnn_25/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02Q
ODouble_Bidirectional_GRU/forward_rnn_25/TensorArrayUnstack/TensorListFromTensorÈ
=Double_Bidirectional_GRU/forward_rnn_25/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=Double_Bidirectional_GRU/forward_rnn_25/strided_slice_2/stackÌ
?Double_Bidirectional_GRU/forward_rnn_25/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2A
?Double_Bidirectional_GRU/forward_rnn_25/strided_slice_2/stack_1Ì
?Double_Bidirectional_GRU/forward_rnn_25/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?Double_Bidirectional_GRU/forward_rnn_25/strided_slice_2/stack_2í
7Double_Bidirectional_GRU/forward_rnn_25/strided_slice_2StridedSlice5Double_Bidirectional_GRU/forward_rnn_25/transpose:y:0FDouble_Bidirectional_GRU/forward_rnn_25/strided_slice_2/stack:output:0HDouble_Bidirectional_GRU/forward_rnn_25/strided_slice_2/stack_1:output:0HDouble_Bidirectional_GRU/forward_rnn_25/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask29
7Double_Bidirectional_GRU/forward_rnn_25/strided_slice_2
BDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/ReadVariableOpReadVariableOpKdouble_bidirectional_gru_forward_rnn_25_gru_cell_25_readvariableop_resource*
_output_shapes
:	*
dtype02D
BDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/ReadVariableOp
;Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/unstackUnpackJDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2=
;Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/unstack«
IDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMul/ReadVariableOpReadVariableOpRdouble_bidirectional_gru_forward_rnn_25_gru_cell_25_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02K
IDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMul/ReadVariableOpÊ
:Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMulMatMul@Double_Bidirectional_GRU/forward_rnn_25/strided_slice_2:output:0QDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2<
:Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMulÄ
;Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/BiasAddBiasAddDDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMul:product:0DDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2=
;Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/BiasAddÕ
CDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2E
CDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/split/split_dimÿ
9Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/splitSplitLDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/split/split_dim:output:0DDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2;
9Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/split±
KDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOpReadVariableOpTdouble_bidirectional_gru_forward_rnn_25_gru_cell_25_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02M
KDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOpÆ
<Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMul_1MatMul6Double_Bidirectional_GRU/forward_rnn_25/zeros:output:0SDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2>
<Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMul_1Ê
=Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/BiasAdd_1BiasAddFDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMul_1:product:0DDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2?
=Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/BiasAdd_1Ë
9Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2;
9Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/ConstÙ
EDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2G
EDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/split_1/split_dimØ
;Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/split_1SplitVFDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/BiasAdd_1:output:0BDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/Const:output:0NDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2=
;Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/split_1¸
7Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/addAddV2BDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/split:output:0DDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ29
7Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/addõ
;Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/SigmoidSigmoid;Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2=
;Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/Sigmoid¼
9Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/add_1AddV2BDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/split:output:1DDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2;
9Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/add_1û
=Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/Sigmoid_1Sigmoid=Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2?
=Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/Sigmoid_1µ
7Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/mulMulADouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/Sigmoid_1:y:0DDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ29
7Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/mul³
9Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/add_2AddV2BDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/split:output:2;Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2;
9Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/add_2î
8Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/TanhTanh=Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:
8Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/Tanh©
9Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/mul_1Mul?Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/Sigmoid:y:06Double_Bidirectional_GRU/forward_rnn_25/zeros:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2;
9Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/mul_1»
9Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2;
9Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/sub/x±
7Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/subSubBDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/sub/x:output:0?Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ29
7Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/sub«
9Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/mul_2Mul;Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/sub:z:0<Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2;
9Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/mul_2°
9Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/add_3AddV2=Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/mul_1:z:0=Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2;
9Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/add_3ß
EDouble_Bidirectional_GRU/forward_rnn_25/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2G
EDouble_Bidirectional_GRU/forward_rnn_25/TensorArrayV2_1/element_shapeØ
7Double_Bidirectional_GRU/forward_rnn_25/TensorArrayV2_1TensorListReserveNDouble_Bidirectional_GRU/forward_rnn_25/TensorArrayV2_1/element_shape:output:0@Double_Bidirectional_GRU/forward_rnn_25/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7Double_Bidirectional_GRU/forward_rnn_25/TensorArrayV2_1
,Double_Bidirectional_GRU/forward_rnn_25/timeConst*
_output_shapes
: *
dtype0*
value	B : 2.
,Double_Bidirectional_GRU/forward_rnn_25/timeÏ
@Double_Bidirectional_GRU/forward_rnn_25/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2B
@Double_Bidirectional_GRU/forward_rnn_25/while/maximum_iterationsº
:Double_Bidirectional_GRU/forward_rnn_25/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2<
:Double_Bidirectional_GRU/forward_rnn_25/while/loop_counterâ	
-Double_Bidirectional_GRU/forward_rnn_25/whileWhileCDouble_Bidirectional_GRU/forward_rnn_25/while/loop_counter:output:0IDouble_Bidirectional_GRU/forward_rnn_25/while/maximum_iterations:output:05Double_Bidirectional_GRU/forward_rnn_25/time:output:0@Double_Bidirectional_GRU/forward_rnn_25/TensorArrayV2_1:handle:06Double_Bidirectional_GRU/forward_rnn_25/zeros:output:0@Double_Bidirectional_GRU/forward_rnn_25/strided_slice_1:output:0_Double_Bidirectional_GRU/forward_rnn_25/TensorArrayUnstack/TensorListFromTensor:output_handle:0Kdouble_bidirectional_gru_forward_rnn_25_gru_cell_25_readvariableop_resourceRdouble_bidirectional_gru_forward_rnn_25_gru_cell_25_matmul_readvariableop_resourceTdouble_bidirectional_gru_forward_rnn_25_gru_cell_25_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*H
body@R>
<Double_Bidirectional_GRU_forward_rnn_25_while_body_387381071*H
cond@R>
<Double_Bidirectional_GRU_forward_rnn_25_while_cond_387381070*9
output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2/
-Double_Bidirectional_GRU/forward_rnn_25/while
XDouble_Bidirectional_GRU/forward_rnn_25/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2Z
XDouble_Bidirectional_GRU/forward_rnn_25/TensorArrayV2Stack/TensorListStack/element_shape
JDouble_Bidirectional_GRU/forward_rnn_25/TensorArrayV2Stack/TensorListStackTensorListStack6Double_Bidirectional_GRU/forward_rnn_25/while:output:3aDouble_Bidirectional_GRU/forward_rnn_25/TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02L
JDouble_Bidirectional_GRU/forward_rnn_25/TensorArrayV2Stack/TensorListStackÑ
=Double_Bidirectional_GRU/forward_rnn_25/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2?
=Double_Bidirectional_GRU/forward_rnn_25/strided_slice_3/stackÌ
?Double_Bidirectional_GRU/forward_rnn_25/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2A
?Double_Bidirectional_GRU/forward_rnn_25/strided_slice_3/stack_1Ì
?Double_Bidirectional_GRU/forward_rnn_25/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?Double_Bidirectional_GRU/forward_rnn_25/strided_slice_3/stack_2
7Double_Bidirectional_GRU/forward_rnn_25/strided_slice_3StridedSliceSDouble_Bidirectional_GRU/forward_rnn_25/TensorArrayV2Stack/TensorListStack:tensor:0FDouble_Bidirectional_GRU/forward_rnn_25/strided_slice_3/stack:output:0HDouble_Bidirectional_GRU/forward_rnn_25/strided_slice_3/stack_1:output:0HDouble_Bidirectional_GRU/forward_rnn_25/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask29
7Double_Bidirectional_GRU/forward_rnn_25/strided_slice_3É
8Double_Bidirectional_GRU/forward_rnn_25/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2:
8Double_Bidirectional_GRU/forward_rnn_25/transpose_1/permÏ
3Double_Bidirectional_GRU/forward_rnn_25/transpose_1	TransposeSDouble_Bidirectional_GRU/forward_rnn_25/TensorArrayV2Stack/TensorListStack:tensor:0ADouble_Bidirectional_GRU/forward_rnn_25/transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ25
3Double_Bidirectional_GRU/forward_rnn_25/transpose_1³
.Double_Bidirectional_GRU/backward_rnn_25/ShapeShape#Embedding_Dropout/dropout/Mul_1:z:0*
T0*
_output_shapes
:20
.Double_Bidirectional_GRU/backward_rnn_25/ShapeÆ
<Double_Bidirectional_GRU/backward_rnn_25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<Double_Bidirectional_GRU/backward_rnn_25/strided_slice/stackÊ
>Double_Bidirectional_GRU/backward_rnn_25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2@
>Double_Bidirectional_GRU/backward_rnn_25/strided_slice/stack_1Ê
>Double_Bidirectional_GRU/backward_rnn_25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>Double_Bidirectional_GRU/backward_rnn_25/strided_slice/stack_2Ø
6Double_Bidirectional_GRU/backward_rnn_25/strided_sliceStridedSlice7Double_Bidirectional_GRU/backward_rnn_25/Shape:output:0EDouble_Bidirectional_GRU/backward_rnn_25/strided_slice/stack:output:0GDouble_Bidirectional_GRU/backward_rnn_25/strided_slice/stack_1:output:0GDouble_Bidirectional_GRU/backward_rnn_25/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask28
6Double_Bidirectional_GRU/backward_rnn_25/strided_slice¯
4Double_Bidirectional_GRU/backward_rnn_25/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :26
4Double_Bidirectional_GRU/backward_rnn_25/zeros/mul/y
2Double_Bidirectional_GRU/backward_rnn_25/zeros/mulMul?Double_Bidirectional_GRU/backward_rnn_25/strided_slice:output:0=Double_Bidirectional_GRU/backward_rnn_25/zeros/mul/y:output:0*
T0*
_output_shapes
: 24
2Double_Bidirectional_GRU/backward_rnn_25/zeros/mul±
5Double_Bidirectional_GRU/backward_rnn_25/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è27
5Double_Bidirectional_GRU/backward_rnn_25/zeros/Less/y
3Double_Bidirectional_GRU/backward_rnn_25/zeros/LessLess6Double_Bidirectional_GRU/backward_rnn_25/zeros/mul:z:0>Double_Bidirectional_GRU/backward_rnn_25/zeros/Less/y:output:0*
T0*
_output_shapes
: 25
3Double_Bidirectional_GRU/backward_rnn_25/zeros/Lessµ
7Double_Bidirectional_GRU/backward_rnn_25/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :29
7Double_Bidirectional_GRU/backward_rnn_25/zeros/packed/1§
5Double_Bidirectional_GRU/backward_rnn_25/zeros/packedPack?Double_Bidirectional_GRU/backward_rnn_25/strided_slice:output:0@Double_Bidirectional_GRU/backward_rnn_25/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:27
5Double_Bidirectional_GRU/backward_rnn_25/zeros/packed±
4Double_Bidirectional_GRU/backward_rnn_25/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    26
4Double_Bidirectional_GRU/backward_rnn_25/zeros/Const
.Double_Bidirectional_GRU/backward_rnn_25/zerosFill>Double_Bidirectional_GRU/backward_rnn_25/zeros/packed:output:0=Double_Bidirectional_GRU/backward_rnn_25/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.Double_Bidirectional_GRU/backward_rnn_25/zerosÇ
7Double_Bidirectional_GRU/backward_rnn_25/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          29
7Double_Bidirectional_GRU/backward_rnn_25/transpose/perm
2Double_Bidirectional_GRU/backward_rnn_25/transpose	Transpose#Embedding_Dropout/dropout/Mul_1:z:0@Double_Bidirectional_GRU/backward_rnn_25/transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ24
2Double_Bidirectional_GRU/backward_rnn_25/transposeÊ
0Double_Bidirectional_GRU/backward_rnn_25/Shape_1Shape6Double_Bidirectional_GRU/backward_rnn_25/transpose:y:0*
T0*
_output_shapes
:22
0Double_Bidirectional_GRU/backward_rnn_25/Shape_1Ê
>Double_Bidirectional_GRU/backward_rnn_25/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2@
>Double_Bidirectional_GRU/backward_rnn_25/strided_slice_1/stackÎ
@Double_Bidirectional_GRU/backward_rnn_25/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2B
@Double_Bidirectional_GRU/backward_rnn_25/strided_slice_1/stack_1Î
@Double_Bidirectional_GRU/backward_rnn_25/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@Double_Bidirectional_GRU/backward_rnn_25/strided_slice_1/stack_2ä
8Double_Bidirectional_GRU/backward_rnn_25/strided_slice_1StridedSlice9Double_Bidirectional_GRU/backward_rnn_25/Shape_1:output:0GDouble_Bidirectional_GRU/backward_rnn_25/strided_slice_1/stack:output:0IDouble_Bidirectional_GRU/backward_rnn_25/strided_slice_1/stack_1:output:0IDouble_Bidirectional_GRU/backward_rnn_25/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2:
8Double_Bidirectional_GRU/backward_rnn_25/strided_slice_1×
DDouble_Bidirectional_GRU/backward_rnn_25/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2F
DDouble_Bidirectional_GRU/backward_rnn_25/TensorArrayV2/element_shapeÖ
6Double_Bidirectional_GRU/backward_rnn_25/TensorArrayV2TensorListReserveMDouble_Bidirectional_GRU/backward_rnn_25/TensorArrayV2/element_shape:output:0ADouble_Bidirectional_GRU/backward_rnn_25/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type028
6Double_Bidirectional_GRU/backward_rnn_25/TensorArrayV2¼
7Double_Bidirectional_GRU/backward_rnn_25/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 29
7Double_Bidirectional_GRU/backward_rnn_25/ReverseV2/axis¯
2Double_Bidirectional_GRU/backward_rnn_25/ReverseV2	ReverseV26Double_Bidirectional_GRU/backward_rnn_25/transpose:y:0@Double_Bidirectional_GRU/backward_rnn_25/ReverseV2/axis:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ24
2Double_Bidirectional_GRU/backward_rnn_25/ReverseV2
^Double_Bidirectional_GRU/backward_rnn_25/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2`
^Double_Bidirectional_GRU/backward_rnn_25/TensorArrayUnstack/TensorListFromTensor/element_shape¡
PDouble_Bidirectional_GRU/backward_rnn_25/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor;Double_Bidirectional_GRU/backward_rnn_25/ReverseV2:output:0gDouble_Bidirectional_GRU/backward_rnn_25/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02R
PDouble_Bidirectional_GRU/backward_rnn_25/TensorArrayUnstack/TensorListFromTensorÊ
>Double_Bidirectional_GRU/backward_rnn_25/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2@
>Double_Bidirectional_GRU/backward_rnn_25/strided_slice_2/stackÎ
@Double_Bidirectional_GRU/backward_rnn_25/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2B
@Double_Bidirectional_GRU/backward_rnn_25/strided_slice_2/stack_1Î
@Double_Bidirectional_GRU/backward_rnn_25/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@Double_Bidirectional_GRU/backward_rnn_25/strided_slice_2/stack_2ó
8Double_Bidirectional_GRU/backward_rnn_25/strided_slice_2StridedSlice6Double_Bidirectional_GRU/backward_rnn_25/transpose:y:0GDouble_Bidirectional_GRU/backward_rnn_25/strided_slice_2/stack:output:0IDouble_Bidirectional_GRU/backward_rnn_25/strided_slice_2/stack_1:output:0IDouble_Bidirectional_GRU/backward_rnn_25/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2:
8Double_Bidirectional_GRU/backward_rnn_25/strided_slice_2
CDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/ReadVariableOpReadVariableOpLdouble_bidirectional_gru_backward_rnn_25_gru_cell_25_readvariableop_resource*
_output_shapes
:	*
dtype02E
CDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/ReadVariableOp
<Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/unstackUnpackKDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2>
<Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/unstack®
JDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMul/ReadVariableOpReadVariableOpSdouble_bidirectional_gru_backward_rnn_25_gru_cell_25_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02L
JDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMul/ReadVariableOpÎ
;Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMulMatMulADouble_Bidirectional_GRU/backward_rnn_25/strided_slice_2:output:0RDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2=
;Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMulÈ
<Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/BiasAddBiasAddEDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMul:product:0EDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2>
<Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/BiasAdd×
DDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2F
DDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/split/split_dim
:Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/splitSplitMDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/split/split_dim:output:0EDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2<
:Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/split´
LDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOpReadVariableOpUdouble_bidirectional_gru_backward_rnn_25_gru_cell_25_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02N
LDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOpÊ
=Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMul_1MatMul7Double_Bidirectional_GRU/backward_rnn_25/zeros:output:0TDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2?
=Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMul_1Î
>Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/BiasAdd_1BiasAddGDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMul_1:product:0EDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2@
>Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/BiasAdd_1Í
:Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2<
:Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/ConstÛ
FDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2H
FDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/split_1/split_dimÝ
<Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/split_1SplitVGDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/BiasAdd_1:output:0CDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/Const:output:0ODouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2>
<Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/split_1¼
8Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/addAddV2CDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/split:output:0EDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:
8Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/addø
<Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/SigmoidSigmoid<Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2>
<Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/SigmoidÀ
:Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/add_1AddV2CDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/split:output:1EDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2<
:Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/add_1þ
>Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/Sigmoid_1Sigmoid>Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2@
>Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/Sigmoid_1¹
8Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/mulMulBDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/Sigmoid_1:y:0EDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:
8Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/mul·
:Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/add_2AddV2CDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/split:output:2<Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2<
:Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/add_2ñ
9Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/TanhTanh>Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2;
9Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/Tanh­
:Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/mul_1Mul@Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/Sigmoid:y:07Double_Bidirectional_GRU/backward_rnn_25/zeros:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2<
:Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/mul_1½
:Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2<
:Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/sub/xµ
8Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/subSubCDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/sub/x:output:0@Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:
8Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/sub¯
:Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/mul_2Mul<Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/sub:z:0=Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2<
:Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/mul_2´
:Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/add_3AddV2>Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/mul_1:z:0>Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2<
:Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/add_3á
FDouble_Bidirectional_GRU/backward_rnn_25/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2H
FDouble_Bidirectional_GRU/backward_rnn_25/TensorArrayV2_1/element_shapeÜ
8Double_Bidirectional_GRU/backward_rnn_25/TensorArrayV2_1TensorListReserveODouble_Bidirectional_GRU/backward_rnn_25/TensorArrayV2_1/element_shape:output:0ADouble_Bidirectional_GRU/backward_rnn_25/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8Double_Bidirectional_GRU/backward_rnn_25/TensorArrayV2_1 
-Double_Bidirectional_GRU/backward_rnn_25/timeConst*
_output_shapes
: *
dtype0*
value	B : 2/
-Double_Bidirectional_GRU/backward_rnn_25/timeÑ
ADouble_Bidirectional_GRU/backward_rnn_25/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2C
ADouble_Bidirectional_GRU/backward_rnn_25/while/maximum_iterations¼
;Double_Bidirectional_GRU/backward_rnn_25/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2=
;Double_Bidirectional_GRU/backward_rnn_25/while/loop_counterð	
.Double_Bidirectional_GRU/backward_rnn_25/whileWhileDDouble_Bidirectional_GRU/backward_rnn_25/while/loop_counter:output:0JDouble_Bidirectional_GRU/backward_rnn_25/while/maximum_iterations:output:06Double_Bidirectional_GRU/backward_rnn_25/time:output:0ADouble_Bidirectional_GRU/backward_rnn_25/TensorArrayV2_1:handle:07Double_Bidirectional_GRU/backward_rnn_25/zeros:output:0ADouble_Bidirectional_GRU/backward_rnn_25/strided_slice_1:output:0`Double_Bidirectional_GRU/backward_rnn_25/TensorArrayUnstack/TensorListFromTensor:output_handle:0Ldouble_bidirectional_gru_backward_rnn_25_gru_cell_25_readvariableop_resourceSdouble_bidirectional_gru_backward_rnn_25_gru_cell_25_matmul_readvariableop_resourceUdouble_bidirectional_gru_backward_rnn_25_gru_cell_25_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*I
bodyAR?
=Double_Bidirectional_GRU_backward_rnn_25_while_body_387381225*I
condAR?
=Double_Bidirectional_GRU_backward_rnn_25_while_cond_387381224*9
output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 20
.Double_Bidirectional_GRU/backward_rnn_25/while
YDouble_Bidirectional_GRU/backward_rnn_25/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2[
YDouble_Bidirectional_GRU/backward_rnn_25/TensorArrayV2Stack/TensorListStack/element_shape
KDouble_Bidirectional_GRU/backward_rnn_25/TensorArrayV2Stack/TensorListStackTensorListStack7Double_Bidirectional_GRU/backward_rnn_25/while:output:3bDouble_Bidirectional_GRU/backward_rnn_25/TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02M
KDouble_Bidirectional_GRU/backward_rnn_25/TensorArrayV2Stack/TensorListStackÓ
>Double_Bidirectional_GRU/backward_rnn_25/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2@
>Double_Bidirectional_GRU/backward_rnn_25/strided_slice_3/stackÎ
@Double_Bidirectional_GRU/backward_rnn_25/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2B
@Double_Bidirectional_GRU/backward_rnn_25/strided_slice_3/stack_1Î
@Double_Bidirectional_GRU/backward_rnn_25/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@Double_Bidirectional_GRU/backward_rnn_25/strided_slice_3/stack_2
8Double_Bidirectional_GRU/backward_rnn_25/strided_slice_3StridedSliceTDouble_Bidirectional_GRU/backward_rnn_25/TensorArrayV2Stack/TensorListStack:tensor:0GDouble_Bidirectional_GRU/backward_rnn_25/strided_slice_3/stack:output:0IDouble_Bidirectional_GRU/backward_rnn_25/strided_slice_3/stack_1:output:0IDouble_Bidirectional_GRU/backward_rnn_25/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2:
8Double_Bidirectional_GRU/backward_rnn_25/strided_slice_3Ë
9Double_Bidirectional_GRU/backward_rnn_25/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2;
9Double_Bidirectional_GRU/backward_rnn_25/transpose_1/permÓ
4Double_Bidirectional_GRU/backward_rnn_25/transpose_1	TransposeTDouble_Bidirectional_GRU/backward_rnn_25/TensorArrayV2Stack/TensorListStack:tensor:0BDouble_Bidirectional_GRU/backward_rnn_25/transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ26
4Double_Bidirectional_GRU/backward_rnn_25/transpose_1
$Double_Bidirectional_GRU/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2&
$Double_Bidirectional_GRU/concat/axis¾
Double_Bidirectional_GRU/concatConcatV2@Double_Bidirectional_GRU/forward_rnn_25/strided_slice_3:output:0ADouble_Bidirectional_GRU/backward_rnn_25/strided_slice_3:output:0-Double_Bidirectional_GRU/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
Double_Bidirectional_GRU/concat¯
 Prediction/MatMul/ReadVariableOpReadVariableOp)prediction_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 Prediction/MatMul/ReadVariableOp¶
Prediction/MatMulMatMul(Double_Bidirectional_GRU/concat:output:0(Prediction/MatMul/ReadVariableOp:value:0*
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
Prediction/SigmoidÓ
IdentityIdentityPrediction/Sigmoid:y:0K^Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMul/ReadVariableOpM^Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOpD^Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/ReadVariableOp/^Double_Bidirectional_GRU/backward_rnn_25/whileJ^Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMul/ReadVariableOpL^Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOpC^Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/ReadVariableOp.^Double_Bidirectional_GRU/forward_rnn_25/while"^Prediction/BiasAdd/ReadVariableOp!^Prediction/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : 2
JDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMul/ReadVariableOpJDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMul/ReadVariableOp2
LDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOpLDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp2
CDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/ReadVariableOpCDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/ReadVariableOp2`
.Double_Bidirectional_GRU/backward_rnn_25/while.Double_Bidirectional_GRU/backward_rnn_25/while2
IDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMul/ReadVariableOpIDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMul/ReadVariableOp2
KDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOpKDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp2
BDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/ReadVariableOpBDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/ReadVariableOp2^
-Double_Bidirectional_GRU/forward_rnn_25/while-Double_Bidirectional_GRU/forward_rnn_25/while2F
!Prediction/BiasAdd/ReadVariableOp!Prediction/BiasAdd/ReadVariableOp2D
 Prediction/MatMul/ReadVariableOp Prediction/MatMul/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¹
while_cond_387378643
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_17
3while_while_cond_387378643___redundant_placeholder07
3while_while_cond_387378643___redundant_placeholder17
3while_while_cond_387378643___redundant_placeholder27
3while_while_cond_387378643___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
¬#
Î
while_body_387378644
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_gru_cell_25_387378666_0:	1
while_gru_cell_25_387378668_0:
1
while_gru_cell_25_387378670_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_gru_cell_25_387378666:	/
while_gru_cell_25_387378668:
/
while_gru_cell_25_387378670:
¢)while/gru_cell_25/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÂ
)while/gru_cell_25/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_25_387378666_0while_gru_cell_25_387378668_0while_gru_cell_25_387378670_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_gru_cell_25_layer_call_and_return_conditional_losses_3873786312+
)while/gru_cell_25/StatefulPartitionedCallö
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_25/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:0*^while/gru_cell_25/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations*^while/gru_cell_25/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0*^while/gru_cell_25/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¹
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/gru_cell_25/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3Ã
while/Identity_4Identity2while/gru_cell_25/StatefulPartitionedCall:output:1*^while/gru_cell_25/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4"<
while_gru_cell_25_387378666while_gru_cell_25_387378666_0"<
while_gru_cell_25_387378668while_gru_cell_25_387378668_0"<
while_gru_cell_25_387378670while_gru_cell_25_387378670_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/gru_cell_25/StatefulPartitionedCall)while/gru_cell_25/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
¬#
Î
while_body_387379422
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_gru_cell_25_387379444_0:	1
while_gru_cell_25_387379446_0:
1
while_gru_cell_25_387379448_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_gru_cell_25_387379444:	/
while_gru_cell_25_387379446:
/
while_gru_cell_25_387379448:
¢)while/gru_cell_25/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÂ
)while/gru_cell_25/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_25_387379444_0while_gru_cell_25_387379446_0while_gru_cell_25_387379448_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_gru_cell_25_layer_call_and_return_conditional_losses_3873793532+
)while/gru_cell_25/StatefulPartitionedCallö
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_25/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:0*^while/gru_cell_25/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations*^while/gru_cell_25/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0*^while/gru_cell_25/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¹
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/gru_cell_25/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3Ã
while/Identity_4Identity2while/gru_cell_25/StatefulPartitionedCall:output:1*^while/gru_cell_25/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4"<
while_gru_cell_25_387379444while_gru_cell_25_387379444_0"<
while_gru_cell_25_387379446while_gru_cell_25_387379446_0"<
while_gru_cell_25_387379448while_gru_cell_25_387379448_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/gru_cell_25/StatefulPartitionedCall)while/gru_cell_25/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Ù
Ã
<Double_Bidirectional_GRU_forward_rnn_25_while_body_387380744l
hdouble_bidirectional_gru_forward_rnn_25_while_double_bidirectional_gru_forward_rnn_25_while_loop_counterr
ndouble_bidirectional_gru_forward_rnn_25_while_double_bidirectional_gru_forward_rnn_25_while_maximum_iterations=
9double_bidirectional_gru_forward_rnn_25_while_placeholder?
;double_bidirectional_gru_forward_rnn_25_while_placeholder_1?
;double_bidirectional_gru_forward_rnn_25_while_placeholder_2k
gdouble_bidirectional_gru_forward_rnn_25_while_double_bidirectional_gru_forward_rnn_25_strided_slice_1_0¨
£double_bidirectional_gru_forward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_double_bidirectional_gru_forward_rnn_25_tensorarrayunstack_tensorlistfromtensor_0f
Sdouble_bidirectional_gru_forward_rnn_25_while_gru_cell_25_readvariableop_resource_0:	n
Zdouble_bidirectional_gru_forward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource_0:
p
\double_bidirectional_gru_forward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource_0:
:
6double_bidirectional_gru_forward_rnn_25_while_identity<
8double_bidirectional_gru_forward_rnn_25_while_identity_1<
8double_bidirectional_gru_forward_rnn_25_while_identity_2<
8double_bidirectional_gru_forward_rnn_25_while_identity_3<
8double_bidirectional_gru_forward_rnn_25_while_identity_4i
edouble_bidirectional_gru_forward_rnn_25_while_double_bidirectional_gru_forward_rnn_25_strided_slice_1¦
¡double_bidirectional_gru_forward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_double_bidirectional_gru_forward_rnn_25_tensorarrayunstack_tensorlistfromtensord
Qdouble_bidirectional_gru_forward_rnn_25_while_gru_cell_25_readvariableop_resource:	l
Xdouble_bidirectional_gru_forward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource:
n
Zdouble_bidirectional_gru_forward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource:
¢ODouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp¢QDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp¢HDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/ReadVariableOp
_Double_Bidirectional_GRU/forward_rnn_25/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2a
_Double_Bidirectional_GRU/forward_rnn_25/while/TensorArrayV2Read/TensorListGetItem/element_shapeÅ
QDouble_Bidirectional_GRU/forward_rnn_25/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem£double_bidirectional_gru_forward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_double_bidirectional_gru_forward_rnn_25_tensorarrayunstack_tensorlistfromtensor_09double_bidirectional_gru_forward_rnn_25_while_placeholderhDouble_Bidirectional_GRU/forward_rnn_25/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02S
QDouble_Bidirectional_GRU/forward_rnn_25/while/TensorArrayV2Read/TensorListGetItem©
HDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/ReadVariableOpReadVariableOpSdouble_bidirectional_gru_forward_rnn_25_while_gru_cell_25_readvariableop_resource_0*
_output_shapes
:	*
dtype02J
HDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/ReadVariableOp
ADouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/unstackUnpackPDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2C
ADouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/unstack¿
ODouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOpReadVariableOpZdouble_bidirectional_gru_forward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype02Q
ODouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOpô
@Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMulMatMulXDouble_Bidirectional_GRU/forward_rnn_25/while/TensorArrayV2Read/TensorListGetItem:item:0WDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2B
@Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMulÜ
ADouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/BiasAddBiasAddJDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul:product:0JDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2C
ADouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/BiasAddá
IDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2K
IDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/split/split_dim
?Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/splitSplitRDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/split/split_dim:output:0JDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2A
?Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/splitÅ
QDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpReadVariableOp\double_bidirectional_gru_forward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02S
QDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpÝ
BDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul_1MatMul;double_bidirectional_gru_forward_rnn_25_while_placeholder_2YDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2D
BDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul_1â
CDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/BiasAdd_1BiasAddLDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul_1:product:0JDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2E
CDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/BiasAdd_1×
?Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2A
?Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/Constå
KDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2M
KDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/split_1/split_dimö
ADouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/split_1SplitVLDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/BiasAdd_1:output:0HDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/Const:output:0TDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2C
ADouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/split_1Ð
=Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/addAddV2HDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/split:output:0JDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2?
=Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/add
ADouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/SigmoidSigmoidADouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2C
ADouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/SigmoidÔ
?Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/add_1AddV2HDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/split:output:1JDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2A
?Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/add_1
CDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/Sigmoid_1SigmoidCDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2E
CDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/Sigmoid_1Í
=Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/mulMulGDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/Sigmoid_1:y:0JDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2?
=Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/mulË
?Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/add_2AddV2HDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/split:output:2ADouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2A
?Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/add_2
>Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/TanhTanhCDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2@
>Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/TanhÀ
?Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/mul_1MulEDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/Sigmoid:y:0;double_bidirectional_gru_forward_rnn_25_while_placeholder_2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2A
?Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/mul_1Ç
?Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2A
?Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/sub/xÉ
=Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/subSubHDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/sub/x:output:0EDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2?
=Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/subÃ
?Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/mul_2MulADouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/sub:z:0BDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2A
?Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/mul_2È
?Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/add_3AddV2CDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/mul_1:z:0CDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2A
?Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/add_3§
RDouble_Bidirectional_GRU/forward_rnn_25/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem;double_bidirectional_gru_forward_rnn_25_while_placeholder_19double_bidirectional_gru_forward_rnn_25_while_placeholderCDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/add_3:z:0*
_output_shapes
: *
element_dtype02T
RDouble_Bidirectional_GRU/forward_rnn_25/while/TensorArrayV2Write/TensorListSetItem¬
3Double_Bidirectional_GRU/forward_rnn_25/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :25
3Double_Bidirectional_GRU/forward_rnn_25/while/add/y
1Double_Bidirectional_GRU/forward_rnn_25/while/addAddV29double_bidirectional_gru_forward_rnn_25_while_placeholder<Double_Bidirectional_GRU/forward_rnn_25/while/add/y:output:0*
T0*
_output_shapes
: 23
1Double_Bidirectional_GRU/forward_rnn_25/while/add°
5Double_Bidirectional_GRU/forward_rnn_25/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :27
5Double_Bidirectional_GRU/forward_rnn_25/while/add_1/y¾
3Double_Bidirectional_GRU/forward_rnn_25/while/add_1AddV2hdouble_bidirectional_gru_forward_rnn_25_while_double_bidirectional_gru_forward_rnn_25_while_loop_counter>Double_Bidirectional_GRU/forward_rnn_25/while/add_1/y:output:0*
T0*
_output_shapes
: 25
3Double_Bidirectional_GRU/forward_rnn_25/while/add_1Ç
6Double_Bidirectional_GRU/forward_rnn_25/while/IdentityIdentity7Double_Bidirectional_GRU/forward_rnn_25/while/add_1:z:0P^Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOpR^Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpI^Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 28
6Double_Bidirectional_GRU/forward_rnn_25/while/Identity
8Double_Bidirectional_GRU/forward_rnn_25/while/Identity_1Identityndouble_bidirectional_gru_forward_rnn_25_while_double_bidirectional_gru_forward_rnn_25_while_maximum_iterationsP^Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOpR^Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpI^Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2:
8Double_Bidirectional_GRU/forward_rnn_25/while/Identity_1É
8Double_Bidirectional_GRU/forward_rnn_25/while/Identity_2Identity5Double_Bidirectional_GRU/forward_rnn_25/while/add:z:0P^Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOpR^Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpI^Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2:
8Double_Bidirectional_GRU/forward_rnn_25/while/Identity_2ö
8Double_Bidirectional_GRU/forward_rnn_25/while/Identity_3IdentitybDouble_Bidirectional_GRU/forward_rnn_25/while/TensorArrayV2Write/TensorListSetItem:output_handle:0P^Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOpR^Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpI^Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2:
8Double_Bidirectional_GRU/forward_rnn_25/while/Identity_3é
8Double_Bidirectional_GRU/forward_rnn_25/while/Identity_4IdentityCDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/add_3:z:0P^Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOpR^Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpI^Double_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:
8Double_Bidirectional_GRU/forward_rnn_25/while/Identity_4"Ð
edouble_bidirectional_gru_forward_rnn_25_while_double_bidirectional_gru_forward_rnn_25_strided_slice_1gdouble_bidirectional_gru_forward_rnn_25_while_double_bidirectional_gru_forward_rnn_25_strided_slice_1_0"º
Zdouble_bidirectional_gru_forward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource\double_bidirectional_gru_forward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource_0"¶
Xdouble_bidirectional_gru_forward_rnn_25_while_gru_cell_25_matmul_readvariableop_resourceZdouble_bidirectional_gru_forward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource_0"¨
Qdouble_bidirectional_gru_forward_rnn_25_while_gru_cell_25_readvariableop_resourceSdouble_bidirectional_gru_forward_rnn_25_while_gru_cell_25_readvariableop_resource_0"y
6double_bidirectional_gru_forward_rnn_25_while_identity?Double_Bidirectional_GRU/forward_rnn_25/while/Identity:output:0"}
8double_bidirectional_gru_forward_rnn_25_while_identity_1ADouble_Bidirectional_GRU/forward_rnn_25/while/Identity_1:output:0"}
8double_bidirectional_gru_forward_rnn_25_while_identity_2ADouble_Bidirectional_GRU/forward_rnn_25/while/Identity_2:output:0"}
8double_bidirectional_gru_forward_rnn_25_while_identity_3ADouble_Bidirectional_GRU/forward_rnn_25/while/Identity_3:output:0"}
8double_bidirectional_gru_forward_rnn_25_while_identity_4ADouble_Bidirectional_GRU/forward_rnn_25/while/Identity_4:output:0"Ê
¡double_bidirectional_gru_forward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_double_bidirectional_gru_forward_rnn_25_tensorarrayunstack_tensorlistfromtensor£double_bidirectional_gru_forward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_double_bidirectional_gru_forward_rnn_25_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : 2¢
ODouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOpODouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp2¦
QDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpQDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp2
HDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/ReadVariableOpHDouble_Bidirectional_GRU/forward_rnn_25/while/gru_cell_25/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
=

M__inference_forward_rnn_25_layer_call_and_return_conditional_losses_387378903

inputs(
gru_cell_25_387378828:	)
gru_cell_25_387378830:
)
gru_cell_25_387378832:

identity¢#gru_cell_25/StatefulPartitionedCall¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2
#gru_cell_25/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_25_387378828gru_cell_25_387378830gru_cell_25_387378832*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_gru_cell_25_layer_call_and_return_conditional_losses_3873787732%
#gru_cell_25/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterø
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_25_387378828gru_cell_25_387378830gru_cell_25_387378832*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	* 
bodyR
while_body_387378840* 
condR
while_cond_387378839*9
output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
transpose_1
IdentityIdentitystrided_slice_3:output:0$^gru_cell_25/StatefulPartitionedCall^while*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2J
#gru_cell_25/StatefulPartitionedCall#gru_cell_25/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Â

à
/__inference_gru_cell_25_layer_call_fn_387382713

inputs
states_0
unknown:	
	unknown_0:

	unknown_1:

identity

identity_1¢StatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_gru_cell_25_layer_call_and_return_conditional_losses_3873786312
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0
?

N__inference_backward_rnn_25_layer_call_and_return_conditional_losses_387379485

inputs(
gru_cell_25_387379410:	)
gru_cell_25_387379412:
)
gru_cell_25_387379414:

identity¢#gru_cell_25/StatefulPartitionedCall¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2j
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2
ReverseV2/axis
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	ReverseV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeý
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2
#gru_cell_25/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_25_387379410gru_cell_25_387379412gru_cell_25_387379414*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_gru_cell_25_layer_call_and_return_conditional_losses_3873793532%
#gru_cell_25/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterø
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_25_387379410gru_cell_25_387379412gru_cell_25_387379414*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	* 
bodyR
while_body_387379422* 
condR
while_cond_387379421*9
output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
transpose_1
IdentityIdentitystrided_slice_3:output:0$^gru_cell_25/StatefulPartitionedCall^while*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2J
#gru_cell_25/StatefulPartitionedCall#gru_cell_25/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ
¡
W__inference_Double_Bidirectional_GRU_layer_call_and_return_conditional_losses_387380047

inputsE
2forward_rnn_25_gru_cell_25_readvariableop_resource:	M
9forward_rnn_25_gru_cell_25_matmul_readvariableop_resource:
O
;forward_rnn_25_gru_cell_25_matmul_1_readvariableop_resource:
F
3backward_rnn_25_gru_cell_25_readvariableop_resource:	N
:backward_rnn_25_gru_cell_25_matmul_readvariableop_resource:
P
<backward_rnn_25_gru_cell_25_matmul_1_readvariableop_resource:

identity¢1backward_rnn_25/gru_cell_25/MatMul/ReadVariableOp¢3backward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp¢*backward_rnn_25/gru_cell_25/ReadVariableOp¢backward_rnn_25/while¢0forward_rnn_25/gru_cell_25/MatMul/ReadVariableOp¢2forward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp¢)forward_rnn_25/gru_cell_25/ReadVariableOp¢forward_rnn_25/whileb
forward_rnn_25/ShapeShapeinputs*
T0*
_output_shapes
:2
forward_rnn_25/Shape
"forward_rnn_25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"forward_rnn_25/strided_slice/stack
$forward_rnn_25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$forward_rnn_25/strided_slice/stack_1
$forward_rnn_25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$forward_rnn_25/strided_slice/stack_2¼
forward_rnn_25/strided_sliceStridedSliceforward_rnn_25/Shape:output:0+forward_rnn_25/strided_slice/stack:output:0-forward_rnn_25/strided_slice/stack_1:output:0-forward_rnn_25/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
forward_rnn_25/strided_slice{
forward_rnn_25/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
forward_rnn_25/zeros/mul/y¨
forward_rnn_25/zeros/mulMul%forward_rnn_25/strided_slice:output:0#forward_rnn_25/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_rnn_25/zeros/mul}
forward_rnn_25/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
forward_rnn_25/zeros/Less/y£
forward_rnn_25/zeros/LessLessforward_rnn_25/zeros/mul:z:0$forward_rnn_25/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_rnn_25/zeros/Less
forward_rnn_25/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
forward_rnn_25/zeros/packed/1¿
forward_rnn_25/zeros/packedPack%forward_rnn_25/strided_slice:output:0&forward_rnn_25/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_rnn_25/zeros/packed}
forward_rnn_25/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_rnn_25/zeros/Const²
forward_rnn_25/zerosFill$forward_rnn_25/zeros/packed:output:0#forward_rnn_25/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
forward_rnn_25/zeros
forward_rnn_25/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
forward_rnn_25/transpose/perm±
forward_rnn_25/transpose	Transposeinputs&forward_rnn_25/transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
forward_rnn_25/transpose|
forward_rnn_25/Shape_1Shapeforward_rnn_25/transpose:y:0*
T0*
_output_shapes
:2
forward_rnn_25/Shape_1
$forward_rnn_25/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_rnn_25/strided_slice_1/stack
&forward_rnn_25/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_rnn_25/strided_slice_1/stack_1
&forward_rnn_25/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_rnn_25/strided_slice_1/stack_2È
forward_rnn_25/strided_slice_1StridedSliceforward_rnn_25/Shape_1:output:0-forward_rnn_25/strided_slice_1/stack:output:0/forward_rnn_25/strided_slice_1/stack_1:output:0/forward_rnn_25/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
forward_rnn_25/strided_slice_1£
*forward_rnn_25/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2,
*forward_rnn_25/TensorArrayV2/element_shapeî
forward_rnn_25/TensorArrayV2TensorListReserve3forward_rnn_25/TensorArrayV2/element_shape:output:0'forward_rnn_25/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
forward_rnn_25/TensorArrayV2Ý
Dforward_rnn_25/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2F
Dforward_rnn_25/TensorArrayUnstack/TensorListFromTensor/element_shape´
6forward_rnn_25/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_rnn_25/transpose:y:0Mforward_rnn_25/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type028
6forward_rnn_25/TensorArrayUnstack/TensorListFromTensor
$forward_rnn_25/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_rnn_25/strided_slice_2/stack
&forward_rnn_25/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_rnn_25/strided_slice_2/stack_1
&forward_rnn_25/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_rnn_25/strided_slice_2/stack_2×
forward_rnn_25/strided_slice_2StridedSliceforward_rnn_25/transpose:y:0-forward_rnn_25/strided_slice_2/stack:output:0/forward_rnn_25/strided_slice_2/stack_1:output:0/forward_rnn_25/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2 
forward_rnn_25/strided_slice_2Ê
)forward_rnn_25/gru_cell_25/ReadVariableOpReadVariableOp2forward_rnn_25_gru_cell_25_readvariableop_resource*
_output_shapes
:	*
dtype02+
)forward_rnn_25/gru_cell_25/ReadVariableOp½
"forward_rnn_25/gru_cell_25/unstackUnpack1forward_rnn_25/gru_cell_25/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2$
"forward_rnn_25/gru_cell_25/unstackà
0forward_rnn_25/gru_cell_25/MatMul/ReadVariableOpReadVariableOp9forward_rnn_25_gru_cell_25_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype022
0forward_rnn_25/gru_cell_25/MatMul/ReadVariableOpæ
!forward_rnn_25/gru_cell_25/MatMulMatMul'forward_rnn_25/strided_slice_2:output:08forward_rnn_25/gru_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!forward_rnn_25/gru_cell_25/MatMulà
"forward_rnn_25/gru_cell_25/BiasAddBiasAdd+forward_rnn_25/gru_cell_25/MatMul:product:0+forward_rnn_25/gru_cell_25/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"forward_rnn_25/gru_cell_25/BiasAdd£
*forward_rnn_25/gru_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2,
*forward_rnn_25/gru_cell_25/split/split_dim
 forward_rnn_25/gru_cell_25/splitSplit3forward_rnn_25/gru_cell_25/split/split_dim:output:0+forward_rnn_25/gru_cell_25/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2"
 forward_rnn_25/gru_cell_25/splitæ
2forward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOpReadVariableOp;forward_rnn_25_gru_cell_25_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype024
2forward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOpâ
#forward_rnn_25/gru_cell_25/MatMul_1MatMulforward_rnn_25/zeros:output:0:forward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#forward_rnn_25/gru_cell_25/MatMul_1æ
$forward_rnn_25/gru_cell_25/BiasAdd_1BiasAdd-forward_rnn_25/gru_cell_25/MatMul_1:product:0+forward_rnn_25/gru_cell_25/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$forward_rnn_25/gru_cell_25/BiasAdd_1
 forward_rnn_25/gru_cell_25/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2"
 forward_rnn_25/gru_cell_25/Const§
,forward_rnn_25/gru_cell_25/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2.
,forward_rnn_25/gru_cell_25/split_1/split_dimÛ
"forward_rnn_25/gru_cell_25/split_1SplitV-forward_rnn_25/gru_cell_25/BiasAdd_1:output:0)forward_rnn_25/gru_cell_25/Const:output:05forward_rnn_25/gru_cell_25/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2$
"forward_rnn_25/gru_cell_25/split_1Ô
forward_rnn_25/gru_cell_25/addAddV2)forward_rnn_25/gru_cell_25/split:output:0+forward_rnn_25/gru_cell_25/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
forward_rnn_25/gru_cell_25/addª
"forward_rnn_25/gru_cell_25/SigmoidSigmoid"forward_rnn_25/gru_cell_25/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"forward_rnn_25/gru_cell_25/SigmoidØ
 forward_rnn_25/gru_cell_25/add_1AddV2)forward_rnn_25/gru_cell_25/split:output:1+forward_rnn_25/gru_cell_25/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 forward_rnn_25/gru_cell_25/add_1°
$forward_rnn_25/gru_cell_25/Sigmoid_1Sigmoid$forward_rnn_25/gru_cell_25/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$forward_rnn_25/gru_cell_25/Sigmoid_1Ñ
forward_rnn_25/gru_cell_25/mulMul(forward_rnn_25/gru_cell_25/Sigmoid_1:y:0+forward_rnn_25/gru_cell_25/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
forward_rnn_25/gru_cell_25/mulÏ
 forward_rnn_25/gru_cell_25/add_2AddV2)forward_rnn_25/gru_cell_25/split:output:2"forward_rnn_25/gru_cell_25/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 forward_rnn_25/gru_cell_25/add_2£
forward_rnn_25/gru_cell_25/TanhTanh$forward_rnn_25/gru_cell_25/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
forward_rnn_25/gru_cell_25/TanhÅ
 forward_rnn_25/gru_cell_25/mul_1Mul&forward_rnn_25/gru_cell_25/Sigmoid:y:0forward_rnn_25/zeros:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 forward_rnn_25/gru_cell_25/mul_1
 forward_rnn_25/gru_cell_25/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2"
 forward_rnn_25/gru_cell_25/sub/xÍ
forward_rnn_25/gru_cell_25/subSub)forward_rnn_25/gru_cell_25/sub/x:output:0&forward_rnn_25/gru_cell_25/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
forward_rnn_25/gru_cell_25/subÇ
 forward_rnn_25/gru_cell_25/mul_2Mul"forward_rnn_25/gru_cell_25/sub:z:0#forward_rnn_25/gru_cell_25/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 forward_rnn_25/gru_cell_25/mul_2Ì
 forward_rnn_25/gru_cell_25/add_3AddV2$forward_rnn_25/gru_cell_25/mul_1:z:0$forward_rnn_25/gru_cell_25/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 forward_rnn_25/gru_cell_25/add_3­
,forward_rnn_25/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2.
,forward_rnn_25/TensorArrayV2_1/element_shapeô
forward_rnn_25/TensorArrayV2_1TensorListReserve5forward_rnn_25/TensorArrayV2_1/element_shape:output:0'forward_rnn_25/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
forward_rnn_25/TensorArrayV2_1l
forward_rnn_25/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_rnn_25/time
'forward_rnn_25/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2)
'forward_rnn_25/while/maximum_iterations
!forward_rnn_25/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2#
!forward_rnn_25/while/loop_counter
forward_rnn_25/whileWhile*forward_rnn_25/while/loop_counter:output:00forward_rnn_25/while/maximum_iterations:output:0forward_rnn_25/time:output:0'forward_rnn_25/TensorArrayV2_1:handle:0forward_rnn_25/zeros:output:0'forward_rnn_25/strided_slice_1:output:0Fforward_rnn_25/TensorArrayUnstack/TensorListFromTensor:output_handle:02forward_rnn_25_gru_cell_25_readvariableop_resource9forward_rnn_25_gru_cell_25_matmul_readvariableop_resource;forward_rnn_25_gru_cell_25_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*/
body'R%
#forward_rnn_25_while_body_387379803*/
cond'R%
#forward_rnn_25_while_cond_387379802*9
output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
forward_rnn_25/whileÓ
?forward_rnn_25/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2A
?forward_rnn_25/TensorArrayV2Stack/TensorListStack/element_shape®
1forward_rnn_25/TensorArrayV2Stack/TensorListStackTensorListStackforward_rnn_25/while:output:3Hforward_rnn_25/TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype023
1forward_rnn_25/TensorArrayV2Stack/TensorListStack
$forward_rnn_25/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2&
$forward_rnn_25/strided_slice_3/stack
&forward_rnn_25/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_rnn_25/strided_slice_3/stack_1
&forward_rnn_25/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_rnn_25/strided_slice_3/stack_2õ
forward_rnn_25/strided_slice_3StridedSlice:forward_rnn_25/TensorArrayV2Stack/TensorListStack:tensor:0-forward_rnn_25/strided_slice_3/stack:output:0/forward_rnn_25/strided_slice_3/stack_1:output:0/forward_rnn_25/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2 
forward_rnn_25/strided_slice_3
forward_rnn_25/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
forward_rnn_25/transpose_1/permë
forward_rnn_25/transpose_1	Transpose:forward_rnn_25/TensorArrayV2Stack/TensorListStack:tensor:0(forward_rnn_25/transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
forward_rnn_25/transpose_1d
backward_rnn_25/ShapeShapeinputs*
T0*
_output_shapes
:2
backward_rnn_25/Shape
#backward_rnn_25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#backward_rnn_25/strided_slice/stack
%backward_rnn_25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%backward_rnn_25/strided_slice/stack_1
%backward_rnn_25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%backward_rnn_25/strided_slice/stack_2Â
backward_rnn_25/strided_sliceStridedSlicebackward_rnn_25/Shape:output:0,backward_rnn_25/strided_slice/stack:output:0.backward_rnn_25/strided_slice/stack_1:output:0.backward_rnn_25/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
backward_rnn_25/strided_slice}
backward_rnn_25/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
backward_rnn_25/zeros/mul/y¬
backward_rnn_25/zeros/mulMul&backward_rnn_25/strided_slice:output:0$backward_rnn_25/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_rnn_25/zeros/mul
backward_rnn_25/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
backward_rnn_25/zeros/Less/y§
backward_rnn_25/zeros/LessLessbackward_rnn_25/zeros/mul:z:0%backward_rnn_25/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_rnn_25/zeros/Less
backward_rnn_25/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2 
backward_rnn_25/zeros/packed/1Ã
backward_rnn_25/zeros/packedPack&backward_rnn_25/strided_slice:output:0'backward_rnn_25/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
backward_rnn_25/zeros/packed
backward_rnn_25/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_rnn_25/zeros/Const¶
backward_rnn_25/zerosFill%backward_rnn_25/zeros/packed:output:0$backward_rnn_25/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
backward_rnn_25/zeros
backward_rnn_25/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
backward_rnn_25/transpose/perm´
backward_rnn_25/transpose	Transposeinputs'backward_rnn_25/transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
backward_rnn_25/transpose
backward_rnn_25/Shape_1Shapebackward_rnn_25/transpose:y:0*
T0*
_output_shapes
:2
backward_rnn_25/Shape_1
%backward_rnn_25/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_rnn_25/strided_slice_1/stack
'backward_rnn_25/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_rnn_25/strided_slice_1/stack_1
'backward_rnn_25/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_rnn_25/strided_slice_1/stack_2Î
backward_rnn_25/strided_slice_1StridedSlice backward_rnn_25/Shape_1:output:0.backward_rnn_25/strided_slice_1/stack:output:00backward_rnn_25/strided_slice_1/stack_1:output:00backward_rnn_25/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
backward_rnn_25/strided_slice_1¥
+backward_rnn_25/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2-
+backward_rnn_25/TensorArrayV2/element_shapeò
backward_rnn_25/TensorArrayV2TensorListReserve4backward_rnn_25/TensorArrayV2/element_shape:output:0(backward_rnn_25/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
backward_rnn_25/TensorArrayV2
backward_rnn_25/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2 
backward_rnn_25/ReverseV2/axisË
backward_rnn_25/ReverseV2	ReverseV2backward_rnn_25/transpose:y:0'backward_rnn_25/ReverseV2/axis:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
backward_rnn_25/ReverseV2ß
Ebackward_rnn_25/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2G
Ebackward_rnn_25/TensorArrayUnstack/TensorListFromTensor/element_shape½
7backward_rnn_25/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"backward_rnn_25/ReverseV2:output:0Nbackward_rnn_25/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7backward_rnn_25/TensorArrayUnstack/TensorListFromTensor
%backward_rnn_25/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_rnn_25/strided_slice_2/stack
'backward_rnn_25/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_rnn_25/strided_slice_2/stack_1
'backward_rnn_25/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_rnn_25/strided_slice_2/stack_2Ý
backward_rnn_25/strided_slice_2StridedSlicebackward_rnn_25/transpose:y:0.backward_rnn_25/strided_slice_2/stack:output:00backward_rnn_25/strided_slice_2/stack_1:output:00backward_rnn_25/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2!
backward_rnn_25/strided_slice_2Í
*backward_rnn_25/gru_cell_25/ReadVariableOpReadVariableOp3backward_rnn_25_gru_cell_25_readvariableop_resource*
_output_shapes
:	*
dtype02,
*backward_rnn_25/gru_cell_25/ReadVariableOpÀ
#backward_rnn_25/gru_cell_25/unstackUnpack2backward_rnn_25/gru_cell_25/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2%
#backward_rnn_25/gru_cell_25/unstackã
1backward_rnn_25/gru_cell_25/MatMul/ReadVariableOpReadVariableOp:backward_rnn_25_gru_cell_25_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype023
1backward_rnn_25/gru_cell_25/MatMul/ReadVariableOpê
"backward_rnn_25/gru_cell_25/MatMulMatMul(backward_rnn_25/strided_slice_2:output:09backward_rnn_25/gru_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"backward_rnn_25/gru_cell_25/MatMulä
#backward_rnn_25/gru_cell_25/BiasAddBiasAdd,backward_rnn_25/gru_cell_25/MatMul:product:0,backward_rnn_25/gru_cell_25/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#backward_rnn_25/gru_cell_25/BiasAdd¥
+backward_rnn_25/gru_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2-
+backward_rnn_25/gru_cell_25/split/split_dim
!backward_rnn_25/gru_cell_25/splitSplit4backward_rnn_25/gru_cell_25/split/split_dim:output:0,backward_rnn_25/gru_cell_25/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2#
!backward_rnn_25/gru_cell_25/splité
3backward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOpReadVariableOp<backward_rnn_25_gru_cell_25_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype025
3backward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOpæ
$backward_rnn_25/gru_cell_25/MatMul_1MatMulbackward_rnn_25/zeros:output:0;backward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$backward_rnn_25/gru_cell_25/MatMul_1ê
%backward_rnn_25/gru_cell_25/BiasAdd_1BiasAdd.backward_rnn_25/gru_cell_25/MatMul_1:product:0,backward_rnn_25/gru_cell_25/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%backward_rnn_25/gru_cell_25/BiasAdd_1
!backward_rnn_25/gru_cell_25/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2#
!backward_rnn_25/gru_cell_25/Const©
-backward_rnn_25/gru_cell_25/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2/
-backward_rnn_25/gru_cell_25/split_1/split_dimà
#backward_rnn_25/gru_cell_25/split_1SplitV.backward_rnn_25/gru_cell_25/BiasAdd_1:output:0*backward_rnn_25/gru_cell_25/Const:output:06backward_rnn_25/gru_cell_25/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2%
#backward_rnn_25/gru_cell_25/split_1Ø
backward_rnn_25/gru_cell_25/addAddV2*backward_rnn_25/gru_cell_25/split:output:0,backward_rnn_25/gru_cell_25/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
backward_rnn_25/gru_cell_25/add­
#backward_rnn_25/gru_cell_25/SigmoidSigmoid#backward_rnn_25/gru_cell_25/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#backward_rnn_25/gru_cell_25/SigmoidÜ
!backward_rnn_25/gru_cell_25/add_1AddV2*backward_rnn_25/gru_cell_25/split:output:1,backward_rnn_25/gru_cell_25/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!backward_rnn_25/gru_cell_25/add_1³
%backward_rnn_25/gru_cell_25/Sigmoid_1Sigmoid%backward_rnn_25/gru_cell_25/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%backward_rnn_25/gru_cell_25/Sigmoid_1Õ
backward_rnn_25/gru_cell_25/mulMul)backward_rnn_25/gru_cell_25/Sigmoid_1:y:0,backward_rnn_25/gru_cell_25/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
backward_rnn_25/gru_cell_25/mulÓ
!backward_rnn_25/gru_cell_25/add_2AddV2*backward_rnn_25/gru_cell_25/split:output:2#backward_rnn_25/gru_cell_25/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!backward_rnn_25/gru_cell_25/add_2¦
 backward_rnn_25/gru_cell_25/TanhTanh%backward_rnn_25/gru_cell_25/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 backward_rnn_25/gru_cell_25/TanhÉ
!backward_rnn_25/gru_cell_25/mul_1Mul'backward_rnn_25/gru_cell_25/Sigmoid:y:0backward_rnn_25/zeros:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!backward_rnn_25/gru_cell_25/mul_1
!backward_rnn_25/gru_cell_25/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!backward_rnn_25/gru_cell_25/sub/xÑ
backward_rnn_25/gru_cell_25/subSub*backward_rnn_25/gru_cell_25/sub/x:output:0'backward_rnn_25/gru_cell_25/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
backward_rnn_25/gru_cell_25/subË
!backward_rnn_25/gru_cell_25/mul_2Mul#backward_rnn_25/gru_cell_25/sub:z:0$backward_rnn_25/gru_cell_25/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!backward_rnn_25/gru_cell_25/mul_2Ð
!backward_rnn_25/gru_cell_25/add_3AddV2%backward_rnn_25/gru_cell_25/mul_1:z:0%backward_rnn_25/gru_cell_25/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!backward_rnn_25/gru_cell_25/add_3¯
-backward_rnn_25/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2/
-backward_rnn_25/TensorArrayV2_1/element_shapeø
backward_rnn_25/TensorArrayV2_1TensorListReserve6backward_rnn_25/TensorArrayV2_1/element_shape:output:0(backward_rnn_25/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
backward_rnn_25/TensorArrayV2_1n
backward_rnn_25/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_rnn_25/time
(backward_rnn_25/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2*
(backward_rnn_25/while/maximum_iterations
"backward_rnn_25/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2$
"backward_rnn_25/while/loop_counter
backward_rnn_25/whileWhile+backward_rnn_25/while/loop_counter:output:01backward_rnn_25/while/maximum_iterations:output:0backward_rnn_25/time:output:0(backward_rnn_25/TensorArrayV2_1:handle:0backward_rnn_25/zeros:output:0(backward_rnn_25/strided_slice_1:output:0Gbackward_rnn_25/TensorArrayUnstack/TensorListFromTensor:output_handle:03backward_rnn_25_gru_cell_25_readvariableop_resource:backward_rnn_25_gru_cell_25_matmul_readvariableop_resource<backward_rnn_25_gru_cell_25_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*0
body(R&
$backward_rnn_25_while_body_387379957*0
cond(R&
$backward_rnn_25_while_cond_387379956*9
output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
backward_rnn_25/whileÕ
@backward_rnn_25/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2B
@backward_rnn_25/TensorArrayV2Stack/TensorListStack/element_shape²
2backward_rnn_25/TensorArrayV2Stack/TensorListStackTensorListStackbackward_rnn_25/while:output:3Ibackward_rnn_25/TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype024
2backward_rnn_25/TensorArrayV2Stack/TensorListStack¡
%backward_rnn_25/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2'
%backward_rnn_25/strided_slice_3/stack
'backward_rnn_25/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_rnn_25/strided_slice_3/stack_1
'backward_rnn_25/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_rnn_25/strided_slice_3/stack_2û
backward_rnn_25/strided_slice_3StridedSlice;backward_rnn_25/TensorArrayV2Stack/TensorListStack:tensor:0.backward_rnn_25/strided_slice_3/stack:output:00backward_rnn_25/strided_slice_3/stack_1:output:00backward_rnn_25/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2!
backward_rnn_25/strided_slice_3
 backward_rnn_25/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 backward_rnn_25/transpose_1/permï
backward_rnn_25/transpose_1	Transpose;backward_rnn_25/TensorArrayV2Stack/TensorListStack:tensor:0)backward_rnn_25/transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
backward_rnn_25/transpose_1\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisÁ
concatConcatV2'forward_rnn_25/strided_slice_3:output:0(backward_rnn_25/strided_slice_3:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
concat¾
IdentityIdentityconcat:output:02^backward_rnn_25/gru_cell_25/MatMul/ReadVariableOp4^backward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp+^backward_rnn_25/gru_cell_25/ReadVariableOp^backward_rnn_25/while1^forward_rnn_25/gru_cell_25/MatMul/ReadVariableOp3^forward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp*^forward_rnn_25/gru_cell_25/ReadVariableOp^forward_rnn_25/while*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2f
1backward_rnn_25/gru_cell_25/MatMul/ReadVariableOp1backward_rnn_25/gru_cell_25/MatMul/ReadVariableOp2j
3backward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp3backward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp2X
*backward_rnn_25/gru_cell_25/ReadVariableOp*backward_rnn_25/gru_cell_25/ReadVariableOp2.
backward_rnn_25/whilebackward_rnn_25/while2d
0forward_rnn_25/gru_cell_25/MatMul/ReadVariableOp0forward_rnn_25/gru_cell_25/MatMul/ReadVariableOp2h
2forward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp2forward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp2V
)forward_rnn_25/gru_cell_25/ReadVariableOp)forward_rnn_25/gru_cell_25/ReadVariableOp2,
forward_rnn_25/whileforward_rnn_25/while:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¤
á
=Double_Bidirectional_GRU_backward_rnn_25_while_body_387381225n
jdouble_bidirectional_gru_backward_rnn_25_while_double_bidirectional_gru_backward_rnn_25_while_loop_countert
pdouble_bidirectional_gru_backward_rnn_25_while_double_bidirectional_gru_backward_rnn_25_while_maximum_iterations>
:double_bidirectional_gru_backward_rnn_25_while_placeholder@
<double_bidirectional_gru_backward_rnn_25_while_placeholder_1@
<double_bidirectional_gru_backward_rnn_25_while_placeholder_2m
idouble_bidirectional_gru_backward_rnn_25_while_double_bidirectional_gru_backward_rnn_25_strided_slice_1_0ª
¥double_bidirectional_gru_backward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_double_bidirectional_gru_backward_rnn_25_tensorarrayunstack_tensorlistfromtensor_0g
Tdouble_bidirectional_gru_backward_rnn_25_while_gru_cell_25_readvariableop_resource_0:	o
[double_bidirectional_gru_backward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource_0:
q
]double_bidirectional_gru_backward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource_0:
;
7double_bidirectional_gru_backward_rnn_25_while_identity=
9double_bidirectional_gru_backward_rnn_25_while_identity_1=
9double_bidirectional_gru_backward_rnn_25_while_identity_2=
9double_bidirectional_gru_backward_rnn_25_while_identity_3=
9double_bidirectional_gru_backward_rnn_25_while_identity_4k
gdouble_bidirectional_gru_backward_rnn_25_while_double_bidirectional_gru_backward_rnn_25_strided_slice_1¨
£double_bidirectional_gru_backward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_double_bidirectional_gru_backward_rnn_25_tensorarrayunstack_tensorlistfromtensore
Rdouble_bidirectional_gru_backward_rnn_25_while_gru_cell_25_readvariableop_resource:	m
Ydouble_bidirectional_gru_backward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource:
o
[double_bidirectional_gru_backward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource:
¢PDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp¢RDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp¢IDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/ReadVariableOp
`Double_Bidirectional_GRU/backward_rnn_25/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2b
`Double_Bidirectional_GRU/backward_rnn_25/while/TensorArrayV2Read/TensorListGetItem/element_shapeË
RDouble_Bidirectional_GRU/backward_rnn_25/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem¥double_bidirectional_gru_backward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_double_bidirectional_gru_backward_rnn_25_tensorarrayunstack_tensorlistfromtensor_0:double_bidirectional_gru_backward_rnn_25_while_placeholderiDouble_Bidirectional_GRU/backward_rnn_25/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02T
RDouble_Bidirectional_GRU/backward_rnn_25/while/TensorArrayV2Read/TensorListGetItem¬
IDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/ReadVariableOpReadVariableOpTdouble_bidirectional_gru_backward_rnn_25_while_gru_cell_25_readvariableop_resource_0*
_output_shapes
:	*
dtype02K
IDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/ReadVariableOp
BDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/unstackUnpackQDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2D
BDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/unstackÂ
PDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOpReadVariableOp[double_bidirectional_gru_backward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype02R
PDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOpø
ADouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMulMatMulYDouble_Bidirectional_GRU/backward_rnn_25/while/TensorArrayV2Read/TensorListGetItem:item:0XDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2C
ADouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMulà
BDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/BiasAddBiasAddKDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul:product:0KDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2D
BDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/BiasAddã
JDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2L
JDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/split/split_dim
@Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/splitSplitSDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/split/split_dim:output:0KDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2B
@Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/splitÈ
RDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpReadVariableOp]double_bidirectional_gru_backward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02T
RDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpá
CDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul_1MatMul<double_bidirectional_gru_backward_rnn_25_while_placeholder_2ZDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2E
CDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul_1æ
DDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/BiasAdd_1BiasAddMDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul_1:product:0KDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2F
DDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/BiasAdd_1Ù
@Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2B
@Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/Constç
LDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2N
LDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/split_1/split_dimû
BDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/split_1SplitVMDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/BiasAdd_1:output:0IDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/Const:output:0UDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2D
BDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/split_1Ô
>Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/addAddV2IDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/split:output:0KDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2@
>Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/add
BDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/SigmoidSigmoidBDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2D
BDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/SigmoidØ
@Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/add_1AddV2IDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/split:output:1KDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2B
@Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/add_1
DDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/Sigmoid_1SigmoidDDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2F
DDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/Sigmoid_1Ñ
>Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/mulMulHDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/Sigmoid_1:y:0KDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2@
>Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/mulÏ
@Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/add_2AddV2IDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/split:output:2BDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2B
@Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/add_2
?Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/TanhTanhDDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2A
?Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/TanhÄ
@Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/mul_1MulFDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/Sigmoid:y:0<double_bidirectional_gru_backward_rnn_25_while_placeholder_2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2B
@Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/mul_1É
@Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2B
@Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/sub/xÍ
>Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/subSubIDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/sub/x:output:0FDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2@
>Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/subÇ
@Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/mul_2MulBDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/sub:z:0CDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2B
@Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/mul_2Ì
@Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/add_3AddV2DDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/mul_1:z:0DDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2B
@Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/add_3¬
SDouble_Bidirectional_GRU/backward_rnn_25/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem<double_bidirectional_gru_backward_rnn_25_while_placeholder_1:double_bidirectional_gru_backward_rnn_25_while_placeholderDDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/add_3:z:0*
_output_shapes
: *
element_dtype02U
SDouble_Bidirectional_GRU/backward_rnn_25/while/TensorArrayV2Write/TensorListSetItem®
4Double_Bidirectional_GRU/backward_rnn_25/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :26
4Double_Bidirectional_GRU/backward_rnn_25/while/add/y
2Double_Bidirectional_GRU/backward_rnn_25/while/addAddV2:double_bidirectional_gru_backward_rnn_25_while_placeholder=Double_Bidirectional_GRU/backward_rnn_25/while/add/y:output:0*
T0*
_output_shapes
: 24
2Double_Bidirectional_GRU/backward_rnn_25/while/add²
6Double_Bidirectional_GRU/backward_rnn_25/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :28
6Double_Bidirectional_GRU/backward_rnn_25/while/add_1/yÃ
4Double_Bidirectional_GRU/backward_rnn_25/while/add_1AddV2jdouble_bidirectional_gru_backward_rnn_25_while_double_bidirectional_gru_backward_rnn_25_while_loop_counter?Double_Bidirectional_GRU/backward_rnn_25/while/add_1/y:output:0*
T0*
_output_shapes
: 26
4Double_Bidirectional_GRU/backward_rnn_25/while/add_1Í
7Double_Bidirectional_GRU/backward_rnn_25/while/IdentityIdentity8Double_Bidirectional_GRU/backward_rnn_25/while/add_1:z:0Q^Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOpS^Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpJ^Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 29
7Double_Bidirectional_GRU/backward_rnn_25/while/Identity
9Double_Bidirectional_GRU/backward_rnn_25/while/Identity_1Identitypdouble_bidirectional_gru_backward_rnn_25_while_double_bidirectional_gru_backward_rnn_25_while_maximum_iterationsQ^Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOpS^Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpJ^Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2;
9Double_Bidirectional_GRU/backward_rnn_25/while/Identity_1Ï
9Double_Bidirectional_GRU/backward_rnn_25/while/Identity_2Identity6Double_Bidirectional_GRU/backward_rnn_25/while/add:z:0Q^Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOpS^Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpJ^Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2;
9Double_Bidirectional_GRU/backward_rnn_25/while/Identity_2ü
9Double_Bidirectional_GRU/backward_rnn_25/while/Identity_3IdentitycDouble_Bidirectional_GRU/backward_rnn_25/while/TensorArrayV2Write/TensorListSetItem:output_handle:0Q^Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOpS^Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpJ^Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2;
9Double_Bidirectional_GRU/backward_rnn_25/while/Identity_3ï
9Double_Bidirectional_GRU/backward_rnn_25/while/Identity_4IdentityDDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/add_3:z:0Q^Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOpS^Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpJ^Double_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2;
9Double_Bidirectional_GRU/backward_rnn_25/while/Identity_4"Ô
gdouble_bidirectional_gru_backward_rnn_25_while_double_bidirectional_gru_backward_rnn_25_strided_slice_1idouble_bidirectional_gru_backward_rnn_25_while_double_bidirectional_gru_backward_rnn_25_strided_slice_1_0"¼
[double_bidirectional_gru_backward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource]double_bidirectional_gru_backward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource_0"¸
Ydouble_bidirectional_gru_backward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource[double_bidirectional_gru_backward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource_0"ª
Rdouble_bidirectional_gru_backward_rnn_25_while_gru_cell_25_readvariableop_resourceTdouble_bidirectional_gru_backward_rnn_25_while_gru_cell_25_readvariableop_resource_0"{
7double_bidirectional_gru_backward_rnn_25_while_identity@Double_Bidirectional_GRU/backward_rnn_25/while/Identity:output:0"
9double_bidirectional_gru_backward_rnn_25_while_identity_1BDouble_Bidirectional_GRU/backward_rnn_25/while/Identity_1:output:0"
9double_bidirectional_gru_backward_rnn_25_while_identity_2BDouble_Bidirectional_GRU/backward_rnn_25/while/Identity_2:output:0"
9double_bidirectional_gru_backward_rnn_25_while_identity_3BDouble_Bidirectional_GRU/backward_rnn_25/while/Identity_3:output:0"
9double_bidirectional_gru_backward_rnn_25_while_identity_4BDouble_Bidirectional_GRU/backward_rnn_25/while/Identity_4:output:0"Î
£double_bidirectional_gru_backward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_double_bidirectional_gru_backward_rnn_25_tensorarrayunstack_tensorlistfromtensor¥double_bidirectional_gru_backward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_double_bidirectional_gru_backward_rnn_25_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : 2¤
PDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOpPDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp2¨
RDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpRDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp2
IDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/ReadVariableOpIDouble_Bidirectional_GRU/backward_rnn_25/while/gru_cell_25/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ß
à
J__inference_gru_cell_25_layer_call_and_return_conditional_losses_387382911

inputs
states_0*
readvariableop_resource:	2
matmul_readvariableop_resource:
4
 matmul_1_readvariableop_resource:

identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpy
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
split/split_dim¯
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	BiasAdd_1c
ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2
Constq
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
split_1/split_dimÔ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanh_
mul_1MulSigmoid:y:0states_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sub[
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_3¡
IdentityIdentity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¥

Identity_1Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0
À
Ö
#forward_rnn_25_while_cond_387381450:
6forward_rnn_25_while_forward_rnn_25_while_loop_counter@
<forward_rnn_25_while_forward_rnn_25_while_maximum_iterations$
 forward_rnn_25_while_placeholder&
"forward_rnn_25_while_placeholder_1&
"forward_rnn_25_while_placeholder_2<
8forward_rnn_25_while_less_forward_rnn_25_strided_slice_1U
Qforward_rnn_25_while_forward_rnn_25_while_cond_387381450___redundant_placeholder0U
Qforward_rnn_25_while_forward_rnn_25_while_cond_387381450___redundant_placeholder1U
Qforward_rnn_25_while_forward_rnn_25_while_cond_387381450___redundant_placeholder2U
Qforward_rnn_25_while_forward_rnn_25_while_cond_387381450___redundant_placeholder3!
forward_rnn_25_while_identity
»
forward_rnn_25/while/LessLess forward_rnn_25_while_placeholder8forward_rnn_25_while_less_forward_rnn_25_strided_slice_1*
T0*
_output_shapes
: 2
forward_rnn_25/while/Less
forward_rnn_25/while/IdentityIdentityforward_rnn_25/while/Less:z:0*
T0
*
_output_shapes
: 2
forward_rnn_25/while/Identity"G
forward_rnn_25_while_identity&forward_rnn_25/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ø_
Ñ
#forward_rnn_25_while_body_387381763:
6forward_rnn_25_while_forward_rnn_25_while_loop_counter@
<forward_rnn_25_while_forward_rnn_25_while_maximum_iterations$
 forward_rnn_25_while_placeholder&
"forward_rnn_25_while_placeholder_1&
"forward_rnn_25_while_placeholder_29
5forward_rnn_25_while_forward_rnn_25_strided_slice_1_0u
qforward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_forward_rnn_25_tensorarrayunstack_tensorlistfromtensor_0M
:forward_rnn_25_while_gru_cell_25_readvariableop_resource_0:	U
Aforward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource_0:
W
Cforward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource_0:
!
forward_rnn_25_while_identity#
forward_rnn_25_while_identity_1#
forward_rnn_25_while_identity_2#
forward_rnn_25_while_identity_3#
forward_rnn_25_while_identity_47
3forward_rnn_25_while_forward_rnn_25_strided_slice_1s
oforward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_forward_rnn_25_tensorarrayunstack_tensorlistfromtensorK
8forward_rnn_25_while_gru_cell_25_readvariableop_resource:	S
?forward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource:
U
Aforward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource:
¢6forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp¢8forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp¢/forward_rnn_25/while/gru_cell_25/ReadVariableOpá
Fforward_rnn_25/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2H
Fforward_rnn_25/while/TensorArrayV2Read/TensorListGetItem/element_shape®
8forward_rnn_25/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemqforward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_forward_rnn_25_tensorarrayunstack_tensorlistfromtensor_0 forward_rnn_25_while_placeholderOforward_rnn_25/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02:
8forward_rnn_25/while/TensorArrayV2Read/TensorListGetItemÞ
/forward_rnn_25/while/gru_cell_25/ReadVariableOpReadVariableOp:forward_rnn_25_while_gru_cell_25_readvariableop_resource_0*
_output_shapes
:	*
dtype021
/forward_rnn_25/while/gru_cell_25/ReadVariableOpÏ
(forward_rnn_25/while/gru_cell_25/unstackUnpack7forward_rnn_25/while/gru_cell_25/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2*
(forward_rnn_25/while/gru_cell_25/unstackô
6forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOpReadVariableOpAforward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype028
6forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp
'forward_rnn_25/while/gru_cell_25/MatMulMatMul?forward_rnn_25/while/TensorArrayV2Read/TensorListGetItem:item:0>forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'forward_rnn_25/while/gru_cell_25/MatMulø
(forward_rnn_25/while/gru_cell_25/BiasAddBiasAdd1forward_rnn_25/while/gru_cell_25/MatMul:product:01forward_rnn_25/while/gru_cell_25/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(forward_rnn_25/while/gru_cell_25/BiasAdd¯
0forward_rnn_25/while/gru_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ22
0forward_rnn_25/while/gru_cell_25/split/split_dim³
&forward_rnn_25/while/gru_cell_25/splitSplit9forward_rnn_25/while/gru_cell_25/split/split_dim:output:01forward_rnn_25/while/gru_cell_25/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2(
&forward_rnn_25/while/gru_cell_25/splitú
8forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpReadVariableOpCforward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02:
8forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpù
)forward_rnn_25/while/gru_cell_25/MatMul_1MatMul"forward_rnn_25_while_placeholder_2@forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)forward_rnn_25/while/gru_cell_25/MatMul_1þ
*forward_rnn_25/while/gru_cell_25/BiasAdd_1BiasAdd3forward_rnn_25/while/gru_cell_25/MatMul_1:product:01forward_rnn_25/while/gru_cell_25/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*forward_rnn_25/while/gru_cell_25/BiasAdd_1¥
&forward_rnn_25/while/gru_cell_25/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2(
&forward_rnn_25/while/gru_cell_25/Const³
2forward_rnn_25/while/gru_cell_25/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ24
2forward_rnn_25/while/gru_cell_25/split_1/split_dimù
(forward_rnn_25/while/gru_cell_25/split_1SplitV3forward_rnn_25/while/gru_cell_25/BiasAdd_1:output:0/forward_rnn_25/while/gru_cell_25/Const:output:0;forward_rnn_25/while/gru_cell_25/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2*
(forward_rnn_25/while/gru_cell_25/split_1ì
$forward_rnn_25/while/gru_cell_25/addAddV2/forward_rnn_25/while/gru_cell_25/split:output:01forward_rnn_25/while/gru_cell_25/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$forward_rnn_25/while/gru_cell_25/add¼
(forward_rnn_25/while/gru_cell_25/SigmoidSigmoid(forward_rnn_25/while/gru_cell_25/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(forward_rnn_25/while/gru_cell_25/Sigmoidð
&forward_rnn_25/while/gru_cell_25/add_1AddV2/forward_rnn_25/while/gru_cell_25/split:output:11forward_rnn_25/while/gru_cell_25/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&forward_rnn_25/while/gru_cell_25/add_1Â
*forward_rnn_25/while/gru_cell_25/Sigmoid_1Sigmoid*forward_rnn_25/while/gru_cell_25/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*forward_rnn_25/while/gru_cell_25/Sigmoid_1é
$forward_rnn_25/while/gru_cell_25/mulMul.forward_rnn_25/while/gru_cell_25/Sigmoid_1:y:01forward_rnn_25/while/gru_cell_25/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$forward_rnn_25/while/gru_cell_25/mulç
&forward_rnn_25/while/gru_cell_25/add_2AddV2/forward_rnn_25/while/gru_cell_25/split:output:2(forward_rnn_25/while/gru_cell_25/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&forward_rnn_25/while/gru_cell_25/add_2µ
%forward_rnn_25/while/gru_cell_25/TanhTanh*forward_rnn_25/while/gru_cell_25/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%forward_rnn_25/while/gru_cell_25/TanhÜ
&forward_rnn_25/while/gru_cell_25/mul_1Mul,forward_rnn_25/while/gru_cell_25/Sigmoid:y:0"forward_rnn_25_while_placeholder_2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&forward_rnn_25/while/gru_cell_25/mul_1
&forward_rnn_25/while/gru_cell_25/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2(
&forward_rnn_25/while/gru_cell_25/sub/xå
$forward_rnn_25/while/gru_cell_25/subSub/forward_rnn_25/while/gru_cell_25/sub/x:output:0,forward_rnn_25/while/gru_cell_25/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$forward_rnn_25/while/gru_cell_25/subß
&forward_rnn_25/while/gru_cell_25/mul_2Mul(forward_rnn_25/while/gru_cell_25/sub:z:0)forward_rnn_25/while/gru_cell_25/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&forward_rnn_25/while/gru_cell_25/mul_2ä
&forward_rnn_25/while/gru_cell_25/add_3AddV2*forward_rnn_25/while/gru_cell_25/mul_1:z:0*forward_rnn_25/while/gru_cell_25/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&forward_rnn_25/while/gru_cell_25/add_3ª
9forward_rnn_25/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem"forward_rnn_25_while_placeholder_1 forward_rnn_25_while_placeholder*forward_rnn_25/while/gru_cell_25/add_3:z:0*
_output_shapes
: *
element_dtype02;
9forward_rnn_25/while/TensorArrayV2Write/TensorListSetItemz
forward_rnn_25/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_rnn_25/while/add/y¥
forward_rnn_25/while/addAddV2 forward_rnn_25_while_placeholder#forward_rnn_25/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_rnn_25/while/add~
forward_rnn_25/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_rnn_25/while/add_1/yÁ
forward_rnn_25/while/add_1AddV26forward_rnn_25_while_forward_rnn_25_while_loop_counter%forward_rnn_25/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_rnn_25/while/add_1±
forward_rnn_25/while/IdentityIdentityforward_rnn_25/while/add_1:z:07^forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp9^forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp0^forward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2
forward_rnn_25/while/IdentityÓ
forward_rnn_25/while/Identity_1Identity<forward_rnn_25_while_forward_rnn_25_while_maximum_iterations7^forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp9^forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp0^forward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2!
forward_rnn_25/while/Identity_1³
forward_rnn_25/while/Identity_2Identityforward_rnn_25/while/add:z:07^forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp9^forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp0^forward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2!
forward_rnn_25/while/Identity_2à
forward_rnn_25/while/Identity_3IdentityIforward_rnn_25/while/TensorArrayV2Write/TensorListSetItem:output_handle:07^forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp9^forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp0^forward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2!
forward_rnn_25/while/Identity_3Ó
forward_rnn_25/while/Identity_4Identity*forward_rnn_25/while/gru_cell_25/add_3:z:07^forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp9^forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp0^forward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
forward_rnn_25/while/Identity_4"l
3forward_rnn_25_while_forward_rnn_25_strided_slice_15forward_rnn_25_while_forward_rnn_25_strided_slice_1_0"
Aforward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resourceCforward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource_0"
?forward_rnn_25_while_gru_cell_25_matmul_readvariableop_resourceAforward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource_0"v
8forward_rnn_25_while_gru_cell_25_readvariableop_resource:forward_rnn_25_while_gru_cell_25_readvariableop_resource_0"G
forward_rnn_25_while_identity&forward_rnn_25/while/Identity:output:0"K
forward_rnn_25_while_identity_1(forward_rnn_25/while/Identity_1:output:0"K
forward_rnn_25_while_identity_2(forward_rnn_25/while/Identity_2:output:0"K
forward_rnn_25_while_identity_3(forward_rnn_25/while/Identity_3:output:0"K
forward_rnn_25_while_identity_4(forward_rnn_25/while/Identity_4:output:0"ä
oforward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_forward_rnn_25_tensorarrayunstack_tensorlistfromtensorqforward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_forward_rnn_25_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : 2p
6forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp6forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp2t
8forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp8forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp2b
/forward_rnn_25/while/gru_cell_25/ReadVariableOp/forward_rnn_25/while/gru_cell_25/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 

Ç
2__inference_forward_rnn_25_layer_call_fn_387382049
inputs_0
unknown:	
	unknown_0:

	unknown_1:

identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_forward_rnn_25_layer_call_and_return_conditional_losses_3873789032
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
Ø
õ
$__inference__wrapped_model_387378557

inputsk
Xtree_encoder_double_bidirectional_gru_forward_rnn_25_gru_cell_25_readvariableop_resource:	s
_tree_encoder_double_bidirectional_gru_forward_rnn_25_gru_cell_25_matmul_readvariableop_resource:
u
atree_encoder_double_bidirectional_gru_forward_rnn_25_gru_cell_25_matmul_1_readvariableop_resource:
l
Ytree_encoder_double_bidirectional_gru_backward_rnn_25_gru_cell_25_readvariableop_resource:	t
`tree_encoder_double_bidirectional_gru_backward_rnn_25_gru_cell_25_matmul_readvariableop_resource:
v
btree_encoder_double_bidirectional_gru_backward_rnn_25_gru_cell_25_matmul_1_readvariableop_resource:
I
6tree_encoder_prediction_matmul_readvariableop_resource:	E
7tree_encoder_prediction_biasadd_readvariableop_resource:
identity¢WTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMul/ReadVariableOp¢YTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp¢PTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/ReadVariableOp¢;Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while¢VTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMul/ReadVariableOp¢XTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp¢OTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/ReadVariableOp¢:Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while¢.Tree_Encoder/Prediction/BiasAdd/ReadVariableOp¢-Tree_Encoder/Prediction/MatMul/ReadVariableOp¦
'Tree_Encoder/Embedding_Dropout/IdentityIdentityinputs*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2)
'Tree_Encoder/Embedding_Dropout/IdentityØ
:Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/ShapeShape0Tree_Encoder/Embedding_Dropout/Identity:output:0*
T0*
_output_shapes
:2<
:Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/ShapeÞ
HTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2J
HTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/strided_slice/stackâ
JTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2L
JTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/strided_slice/stack_1â
JTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2L
JTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/strided_slice/stack_2 
BTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/strided_sliceStridedSliceCTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/Shape:output:0QTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/strided_slice/stack:output:0STree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/strided_slice/stack_1:output:0STree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2D
BTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/strided_sliceÇ
@Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2B
@Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/zeros/mul/yÀ
>Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/zeros/mulMulKTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/strided_slice:output:0ITree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/zeros/mul/y:output:0*
T0*
_output_shapes
: 2@
>Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/zeros/mulÉ
ATree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2C
ATree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/zeros/Less/y»
?Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/zeros/LessLessBTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/zeros/mul:z:0JTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/zeros/Less/y:output:0*
T0*
_output_shapes
: 2A
?Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/zeros/LessÍ
CTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2E
CTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/zeros/packed/1×
ATree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/zeros/packedPackKTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/strided_slice:output:0LTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2C
ATree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/zeros/packedÉ
@Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2B
@Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/zeros/ConstÊ
:Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/zerosFillJTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/zeros/packed:output:0ITree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2<
:Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/zerosß
CTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2E
CTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/transpose/permÍ
>Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/transpose	Transpose0Tree_Encoder/Embedding_Dropout/Identity:output:0LTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2@
>Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/transposeî
<Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/Shape_1ShapeBTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/transpose:y:0*
T0*
_output_shapes
:2>
<Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/Shape_1â
JTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2L
JTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/strided_slice_1/stackæ
LTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2N
LTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/strided_slice_1/stack_1æ
LTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2N
LTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/strided_slice_1/stack_2¬
DTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/strided_slice_1StridedSliceETree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/Shape_1:output:0STree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/strided_slice_1/stack:output:0UTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/strided_slice_1/stack_1:output:0UTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2F
DTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/strided_slice_1ï
PTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2R
PTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/TensorArrayV2/element_shape
BTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/TensorArrayV2TensorListReserveYTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/TensorArrayV2/element_shape:output:0MTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02D
BTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/TensorArrayV2©
jTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2l
jTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/TensorArrayUnstack/TensorListFromTensor/element_shapeÌ
\Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorBTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/transpose:y:0sTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02^
\Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/TensorArrayUnstack/TensorListFromTensorâ
JTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2L
JTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/strided_slice_2/stackæ
LTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2N
LTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/strided_slice_2/stack_1æ
LTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2N
LTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/strided_slice_2/stack_2»
DTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/strided_slice_2StridedSliceBTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/transpose:y:0STree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/strided_slice_2/stack:output:0UTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/strided_slice_2/stack_1:output:0UTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2F
DTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/strided_slice_2¼
OTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/ReadVariableOpReadVariableOpXtree_encoder_double_bidirectional_gru_forward_rnn_25_gru_cell_25_readvariableop_resource*
_output_shapes
:	*
dtype02Q
OTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/ReadVariableOp¯
HTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/unstackUnpackWTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2J
HTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/unstackÒ
VTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMul/ReadVariableOpReadVariableOp_tree_encoder_double_bidirectional_gru_forward_rnn_25_gru_cell_25_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02X
VTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMul/ReadVariableOpþ
GTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMulMatMulMTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/strided_slice_2:output:0^Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2I
GTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMulø
HTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/BiasAddBiasAddQTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMul:product:0QTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2J
HTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/BiasAddï
PTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2R
PTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/split/split_dim³
FTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/splitSplitYTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/split/split_dim:output:0QTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2H
FTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/splitØ
XTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOpReadVariableOpatree_encoder_double_bidirectional_gru_forward_rnn_25_gru_cell_25_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02Z
XTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOpú
ITree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMul_1MatMulCTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/zeros:output:0`Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2K
ITree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMul_1þ
JTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/BiasAdd_1BiasAddSTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMul_1:product:0QTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2L
JTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/BiasAdd_1å
FTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2H
FTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/Constó
RTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2T
RTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/split_1/split_dim
HTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/split_1SplitVSTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/BiasAdd_1:output:0OTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/Const:output:0[Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2J
HTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/split_1ì
DTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/addAddV2OTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/split:output:0QTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2F
DTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/add
HTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/SigmoidSigmoidHTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2J
HTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/Sigmoidð
FTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/add_1AddV2OTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/split:output:1QTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2H
FTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/add_1¢
JTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/Sigmoid_1SigmoidJTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2L
JTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/Sigmoid_1é
DTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/mulMulNTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/Sigmoid_1:y:0QTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2F
DTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/mulç
FTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/add_2AddV2OTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/split:output:2HTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2H
FTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/add_2
ETree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/TanhTanhJTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2G
ETree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/TanhÝ
FTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/mul_1MulLTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/Sigmoid:y:0CTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/zeros:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2H
FTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/mul_1Õ
FTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2H
FTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/sub/xå
DTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/subSubOTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/sub/x:output:0LTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2F
DTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/subß
FTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/mul_2MulHTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/sub:z:0ITree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2H
FTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/mul_2ä
FTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/add_3AddV2JTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/mul_1:z:0JTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2H
FTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/add_3ù
RTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2T
RTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/TensorArrayV2_1/element_shape
DTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/TensorArrayV2_1TensorListReserve[Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/TensorArrayV2_1/element_shape:output:0MTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02F
DTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/TensorArrayV2_1¸
9Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/timeConst*
_output_shapes
: *
dtype0*
value	B : 2;
9Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/timeé
MTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2O
MTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/maximum_iterationsÔ
GTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2I
GTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/loop_counter
:Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/whileWhilePTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/loop_counter:output:0VTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/maximum_iterations:output:0BTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/time:output:0MTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/TensorArrayV2_1:handle:0CTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/zeros:output:0MTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/strided_slice_1:output:0lTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/TensorArrayUnstack/TensorListFromTensor:output_handle:0Xtree_encoder_double_bidirectional_gru_forward_rnn_25_gru_cell_25_readvariableop_resource_tree_encoder_double_bidirectional_gru_forward_rnn_25_gru_cell_25_matmul_readvariableop_resourceatree_encoder_double_bidirectional_gru_forward_rnn_25_gru_cell_25_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*U
bodyMRK
ITree_Encoder_Double_Bidirectional_GRU_forward_rnn_25_while_body_387378306*U
condMRK
ITree_Encoder_Double_Bidirectional_GRU_forward_rnn_25_while_cond_387378305*9
output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2<
:Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while
eTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2g
eTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/TensorArrayV2Stack/TensorListStack/element_shapeÆ
WTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/TensorArrayV2Stack/TensorListStackTensorListStackCTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while:output:3nTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02Y
WTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/TensorArrayV2Stack/TensorListStackë
JTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2L
JTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/strided_slice_3/stackæ
LTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2N
LTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/strided_slice_3/stack_1æ
LTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2N
LTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/strided_slice_3/stack_2Ù
DTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/strided_slice_3StridedSlice`Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/TensorArrayV2Stack/TensorListStack:tensor:0STree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/strided_slice_3/stack:output:0UTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/strided_slice_3/stack_1:output:0UTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2F
DTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/strided_slice_3ã
ETree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2G
ETree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/transpose_1/perm
@Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/transpose_1	Transpose`Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/TensorArrayV2Stack/TensorListStack:tensor:0NTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2B
@Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/transpose_1Ú
;Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/ShapeShape0Tree_Encoder/Embedding_Dropout/Identity:output:0*
T0*
_output_shapes
:2=
;Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/Shapeà
ITree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2K
ITree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/strided_slice/stackä
KTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2M
KTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/strided_slice/stack_1ä
KTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2M
KTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/strided_slice/stack_2¦
CTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/strided_sliceStridedSliceDTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/Shape:output:0RTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/strided_slice/stack:output:0TTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/strided_slice/stack_1:output:0TTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2E
CTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/strided_sliceÉ
ATree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2C
ATree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/zeros/mul/yÄ
?Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/zeros/mulMulLTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/strided_slice:output:0JTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/zeros/mul/y:output:0*
T0*
_output_shapes
: 2A
?Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/zeros/mulË
BTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2D
BTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/zeros/Less/y¿
@Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/zeros/LessLessCTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/zeros/mul:z:0KTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/zeros/Less/y:output:0*
T0*
_output_shapes
: 2B
@Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/zeros/LessÏ
DTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2F
DTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/zeros/packed/1Û
BTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/zeros/packedPackLTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/strided_slice:output:0MTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2D
BTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/zeros/packedË
ATree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2C
ATree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/zeros/ConstÎ
;Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/zerosFillKTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/zeros/packed:output:0JTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2=
;Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/zerosá
DTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2F
DTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/transpose/permÐ
?Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/transpose	Transpose0Tree_Encoder/Embedding_Dropout/Identity:output:0MTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2A
?Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/transposeñ
=Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/Shape_1ShapeCTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/transpose:y:0*
T0*
_output_shapes
:2?
=Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/Shape_1ä
KTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2M
KTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/strided_slice_1/stackè
MTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2O
MTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/strided_slice_1/stack_1è
MTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2O
MTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/strided_slice_1/stack_2²
ETree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/strided_slice_1StridedSliceFTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/Shape_1:output:0TTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/strided_slice_1/stack:output:0VTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/strided_slice_1/stack_1:output:0VTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2G
ETree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/strided_slice_1ñ
QTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2S
QTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/TensorArrayV2/element_shape
CTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/TensorArrayV2TensorListReserveZTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/TensorArrayV2/element_shape:output:0NTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02E
CTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/TensorArrayV2Ö
DTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2F
DTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/ReverseV2/axisã
?Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/ReverseV2	ReverseV2CTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/transpose:y:0MTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/ReverseV2/axis:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2A
?Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/ReverseV2«
kTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2m
kTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/TensorArrayUnstack/TensorListFromTensor/element_shapeÕ
]Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorHTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/ReverseV2:output:0tTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02_
]Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/TensorArrayUnstack/TensorListFromTensorä
KTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2M
KTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/strided_slice_2/stackè
MTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2O
MTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/strided_slice_2/stack_1è
MTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2O
MTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/strided_slice_2/stack_2Á
ETree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/strided_slice_2StridedSliceCTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/transpose:y:0TTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/strided_slice_2/stack:output:0VTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/strided_slice_2/stack_1:output:0VTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2G
ETree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/strided_slice_2¿
PTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/ReadVariableOpReadVariableOpYtree_encoder_double_bidirectional_gru_backward_rnn_25_gru_cell_25_readvariableop_resource*
_output_shapes
:	*
dtype02R
PTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/ReadVariableOp²
ITree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/unstackUnpackXTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2K
ITree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/unstackÕ
WTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMul/ReadVariableOpReadVariableOp`tree_encoder_double_bidirectional_gru_backward_rnn_25_gru_cell_25_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02Y
WTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMul/ReadVariableOp
HTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMulMatMulNTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/strided_slice_2:output:0_Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2J
HTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMulü
ITree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/BiasAddBiasAddRTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMul:product:0RTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2K
ITree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/BiasAddñ
QTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2S
QTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/split/split_dim·
GTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/splitSplitZTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/split/split_dim:output:0RTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2I
GTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/splitÛ
YTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOpReadVariableOpbtree_encoder_double_bidirectional_gru_backward_rnn_25_gru_cell_25_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02[
YTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOpþ
JTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMul_1MatMulDTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/zeros:output:0aTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2L
JTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMul_1
KTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/BiasAdd_1BiasAddTTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMul_1:product:0RTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2M
KTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/BiasAdd_1ç
GTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2I
GTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/Constõ
STree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2U
STree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/split_1/split_dim
ITree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/split_1SplitVTTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/BiasAdd_1:output:0PTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/Const:output:0\Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2K
ITree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/split_1ð
ETree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/addAddV2PTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/split:output:0RTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2G
ETree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/add
ITree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/SigmoidSigmoidITree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2K
ITree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/Sigmoidô
GTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/add_1AddV2PTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/split:output:1RTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2I
GTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/add_1¥
KTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/Sigmoid_1SigmoidKTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2M
KTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/Sigmoid_1í
ETree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/mulMulOTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/Sigmoid_1:y:0RTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2G
ETree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/mulë
GTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/add_2AddV2PTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/split:output:2ITree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2I
GTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/add_2
FTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/TanhTanhKTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2H
FTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/Tanhá
GTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/mul_1MulMTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/Sigmoid:y:0DTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/zeros:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2I
GTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/mul_1×
GTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2I
GTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/sub/xé
ETree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/subSubPTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/sub/x:output:0MTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2G
ETree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/subã
GTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/mul_2MulITree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/sub:z:0JTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2I
GTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/mul_2è
GTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/add_3AddV2KTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/mul_1:z:0KTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2I
GTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/add_3û
STree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2U
STree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/TensorArrayV2_1/element_shape
ETree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/TensorArrayV2_1TensorListReserve\Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/TensorArrayV2_1/element_shape:output:0NTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02G
ETree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/TensorArrayV2_1º
:Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/timeConst*
_output_shapes
: *
dtype0*
value	B : 2<
:Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/timeë
NTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2P
NTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/maximum_iterationsÖ
HTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2J
HTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/loop_counter¦
;Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/whileWhileQTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/loop_counter:output:0WTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/maximum_iterations:output:0CTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/time:output:0NTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/TensorArrayV2_1:handle:0DTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/zeros:output:0NTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/strided_slice_1:output:0mTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/TensorArrayUnstack/TensorListFromTensor:output_handle:0Ytree_encoder_double_bidirectional_gru_backward_rnn_25_gru_cell_25_readvariableop_resource`tree_encoder_double_bidirectional_gru_backward_rnn_25_gru_cell_25_matmul_readvariableop_resourcebtree_encoder_double_bidirectional_gru_backward_rnn_25_gru_cell_25_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*V
bodyNRL
JTree_Encoder_Double_Bidirectional_GRU_backward_rnn_25_while_body_387378460*V
condNRL
JTree_Encoder_Double_Bidirectional_GRU_backward_rnn_25_while_cond_387378459*9
output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2=
;Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while¡
fTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2h
fTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/TensorArrayV2Stack/TensorListStack/element_shapeÊ
XTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/TensorArrayV2Stack/TensorListStackTensorListStackDTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while:output:3oTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02Z
XTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/TensorArrayV2Stack/TensorListStackí
KTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2M
KTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/strided_slice_3/stackè
MTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2O
MTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/strided_slice_3/stack_1è
MTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2O
MTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/strided_slice_3/stack_2ß
ETree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/strided_slice_3StridedSliceaTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/TensorArrayV2Stack/TensorListStack:tensor:0TTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/strided_slice_3/stack:output:0VTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/strided_slice_3/stack_1:output:0VTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2G
ETree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/strided_slice_3å
FTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2H
FTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/transpose_1/perm
ATree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/transpose_1	TransposeaTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/TensorArrayV2Stack/TensorListStack:tensor:0OTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2C
ATree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/transpose_1¨
1Tree_Encoder/Double_Bidirectional_GRU/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :23
1Tree_Encoder/Double_Bidirectional_GRU/concat/axisÿ
,Tree_Encoder/Double_Bidirectional_GRU/concatConcatV2MTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/strided_slice_3:output:0NTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/strided_slice_3:output:0:Tree_Encoder/Double_Bidirectional_GRU/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,Tree_Encoder/Double_Bidirectional_GRU/concatÖ
-Tree_Encoder/Prediction/MatMul/ReadVariableOpReadVariableOp6tree_encoder_prediction_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02/
-Tree_Encoder/Prediction/MatMul/ReadVariableOpê
Tree_Encoder/Prediction/MatMulMatMul5Tree_Encoder/Double_Bidirectional_GRU/concat:output:05Tree_Encoder/Prediction/MatMul/ReadVariableOp:value:0*
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
Tree_Encoder/Prediction/Sigmoidâ
IdentityIdentity#Tree_Encoder/Prediction/Sigmoid:y:0X^Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMul/ReadVariableOpZ^Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOpQ^Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/ReadVariableOp<^Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/whileW^Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMul/ReadVariableOpY^Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOpP^Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/ReadVariableOp;^Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while/^Tree_Encoder/Prediction/BiasAdd/ReadVariableOp.^Tree_Encoder/Prediction/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : 2²
WTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMul/ReadVariableOpWTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMul/ReadVariableOp2¶
YTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOpYTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp2¤
PTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/ReadVariableOpPTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/ReadVariableOp2z
;Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while;Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while2°
VTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMul/ReadVariableOpVTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMul/ReadVariableOp2´
XTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOpXTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp2¢
OTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/ReadVariableOpOTree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/ReadVariableOp2x
:Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while:Tree_Encoder/Double_Bidirectional_GRU/forward_rnn_25/while2`
.Tree_Encoder/Prediction/BiasAdd/ReadVariableOp.Tree_Encoder/Prediction/BiasAdd/ReadVariableOp2^
-Tree_Encoder/Prediction/MatMul/ReadVariableOp-Tree_Encoder/Prediction/MatMul/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameInputs

¹
while_cond_387378839
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_17
3while_while_cond_387378839___redundant_placeholder07
3while_while_cond_387378839___redundant_placeholder17
3while_while_cond_387378839___redundant_placeholder27
3while_while_cond_387378839___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
¾

K__inference_Tree_Encoder_layer_call_and_return_conditional_losses_387380524

inputs5
"double_bidirectional_gru_387380505:	6
"double_bidirectional_gru_387380507:
6
"double_bidirectional_gru_387380509:
5
"double_bidirectional_gru_387380511:	6
"double_bidirectional_gru_387380513:
6
"double_bidirectional_gru_387380515:
'
prediction_387380518:	"
prediction_387380520:
identity¢0Double_Bidirectional_GRU/StatefulPartitionedCall¢)Embedding_Dropout/StatefulPartitionedCall¢"Prediction/StatefulPartitionedCall
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
GPU2*0J 8 *Y
fTRR
P__inference_Embedding_Dropout_layer_call_and_return_conditional_losses_3873804752+
)Embedding_Dropout/StatefulPartitionedCallµ
0Double_Bidirectional_GRU/StatefulPartitionedCallStatefulPartitionedCall2Embedding_Dropout/StatefulPartitionedCall:output:0"double_bidirectional_gru_387380505"double_bidirectional_gru_387380507"double_bidirectional_gru_387380509"double_bidirectional_gru_387380511"double_bidirectional_gru_387380513"double_bidirectional_gru_387380515*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *`
f[RY
W__inference_Double_Bidirectional_GRU_layer_call_and_return_conditional_losses_38738044022
0Double_Bidirectional_GRU/StatefulPartitionedCallÝ
"Prediction/StatefulPartitionedCallStatefulPartitionedCall9Double_Bidirectional_GRU/StatefulPartitionedCall:output:0prediction_387380518prediction_387380520*
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
GPU2*0J 8 *R
fMRK
I__inference_Prediction_layer_call_and_return_conditional_losses_3873800722$
"Prediction/StatefulPartitionedCall
IdentityIdentity+Prediction/StatefulPartitionedCall:output:01^Double_Bidirectional_GRU/StatefulPartitionedCall*^Embedding_Dropout/StatefulPartitionedCall#^Prediction/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : 2d
0Double_Bidirectional_GRU/StatefulPartitionedCall0Double_Bidirectional_GRU/StatefulPartitionedCall2V
)Embedding_Dropout/StatefulPartitionedCall)Embedding_Dropout/StatefulPartitionedCall2H
"Prediction/StatefulPartitionedCall"Prediction/StatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Q
5__inference_Embedding_Dropout_layer_call_fn_387381327

inputs
identityß
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
GPU2*0J 8 *Y
fTRR
P__inference_Embedding_Dropout_layer_call_and_return_conditional_losses_3873797332
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
Û	
Ö
0__inference_Tree_Encoder_layer_call_fn_387380564

inputs
unknown:	
	unknown_0:

	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:

	unknown_5:	
	unknown_6:
identity¢StatefulPartitionedCallÌ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_Tree_Encoder_layer_call_and_return_conditional_losses_3873805242
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameInputs
Ü
²
K__inference_Tree_Encoder_layer_call_and_return_conditional_losses_387380995

inputs^
Kdouble_bidirectional_gru_forward_rnn_25_gru_cell_25_readvariableop_resource:	f
Rdouble_bidirectional_gru_forward_rnn_25_gru_cell_25_matmul_readvariableop_resource:
h
Tdouble_bidirectional_gru_forward_rnn_25_gru_cell_25_matmul_1_readvariableop_resource:
_
Ldouble_bidirectional_gru_backward_rnn_25_gru_cell_25_readvariableop_resource:	g
Sdouble_bidirectional_gru_backward_rnn_25_gru_cell_25_matmul_readvariableop_resource:
i
Udouble_bidirectional_gru_backward_rnn_25_gru_cell_25_matmul_1_readvariableop_resource:
<
)prediction_matmul_readvariableop_resource:	8
*prediction_biasadd_readvariableop_resource:
identity¢JDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMul/ReadVariableOp¢LDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp¢CDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/ReadVariableOp¢.Double_Bidirectional_GRU/backward_rnn_25/while¢IDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMul/ReadVariableOp¢KDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp¢BDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/ReadVariableOp¢-Double_Bidirectional_GRU/forward_rnn_25/while¢!Prediction/BiasAdd/ReadVariableOp¢ Prediction/MatMul/ReadVariableOp
Embedding_Dropout/IdentityIdentityinputs*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Embedding_Dropout/Identity±
-Double_Bidirectional_GRU/forward_rnn_25/ShapeShape#Embedding_Dropout/Identity:output:0*
T0*
_output_shapes
:2/
-Double_Bidirectional_GRU/forward_rnn_25/ShapeÄ
;Double_Bidirectional_GRU/forward_rnn_25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2=
;Double_Bidirectional_GRU/forward_rnn_25/strided_slice/stackÈ
=Double_Bidirectional_GRU/forward_rnn_25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2?
=Double_Bidirectional_GRU/forward_rnn_25/strided_slice/stack_1È
=Double_Bidirectional_GRU/forward_rnn_25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=Double_Bidirectional_GRU/forward_rnn_25/strided_slice/stack_2Ò
5Double_Bidirectional_GRU/forward_rnn_25/strided_sliceStridedSlice6Double_Bidirectional_GRU/forward_rnn_25/Shape:output:0DDouble_Bidirectional_GRU/forward_rnn_25/strided_slice/stack:output:0FDouble_Bidirectional_GRU/forward_rnn_25/strided_slice/stack_1:output:0FDouble_Bidirectional_GRU/forward_rnn_25/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask27
5Double_Bidirectional_GRU/forward_rnn_25/strided_slice­
3Double_Bidirectional_GRU/forward_rnn_25/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :25
3Double_Bidirectional_GRU/forward_rnn_25/zeros/mul/y
1Double_Bidirectional_GRU/forward_rnn_25/zeros/mulMul>Double_Bidirectional_GRU/forward_rnn_25/strided_slice:output:0<Double_Bidirectional_GRU/forward_rnn_25/zeros/mul/y:output:0*
T0*
_output_shapes
: 23
1Double_Bidirectional_GRU/forward_rnn_25/zeros/mul¯
4Double_Bidirectional_GRU/forward_rnn_25/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è26
4Double_Bidirectional_GRU/forward_rnn_25/zeros/Less/y
2Double_Bidirectional_GRU/forward_rnn_25/zeros/LessLess5Double_Bidirectional_GRU/forward_rnn_25/zeros/mul:z:0=Double_Bidirectional_GRU/forward_rnn_25/zeros/Less/y:output:0*
T0*
_output_shapes
: 24
2Double_Bidirectional_GRU/forward_rnn_25/zeros/Less³
6Double_Bidirectional_GRU/forward_rnn_25/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :28
6Double_Bidirectional_GRU/forward_rnn_25/zeros/packed/1£
4Double_Bidirectional_GRU/forward_rnn_25/zeros/packedPack>Double_Bidirectional_GRU/forward_rnn_25/strided_slice:output:0?Double_Bidirectional_GRU/forward_rnn_25/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:26
4Double_Bidirectional_GRU/forward_rnn_25/zeros/packed¯
3Double_Bidirectional_GRU/forward_rnn_25/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    25
3Double_Bidirectional_GRU/forward_rnn_25/zeros/Const
-Double_Bidirectional_GRU/forward_rnn_25/zerosFill=Double_Bidirectional_GRU/forward_rnn_25/zeros/packed:output:0<Double_Bidirectional_GRU/forward_rnn_25/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-Double_Bidirectional_GRU/forward_rnn_25/zerosÅ
6Double_Bidirectional_GRU/forward_rnn_25/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          28
6Double_Bidirectional_GRU/forward_rnn_25/transpose/perm
1Double_Bidirectional_GRU/forward_rnn_25/transpose	Transpose#Embedding_Dropout/Identity:output:0?Double_Bidirectional_GRU/forward_rnn_25/transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ23
1Double_Bidirectional_GRU/forward_rnn_25/transposeÇ
/Double_Bidirectional_GRU/forward_rnn_25/Shape_1Shape5Double_Bidirectional_GRU/forward_rnn_25/transpose:y:0*
T0*
_output_shapes
:21
/Double_Bidirectional_GRU/forward_rnn_25/Shape_1È
=Double_Bidirectional_GRU/forward_rnn_25/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=Double_Bidirectional_GRU/forward_rnn_25/strided_slice_1/stackÌ
?Double_Bidirectional_GRU/forward_rnn_25/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2A
?Double_Bidirectional_GRU/forward_rnn_25/strided_slice_1/stack_1Ì
?Double_Bidirectional_GRU/forward_rnn_25/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?Double_Bidirectional_GRU/forward_rnn_25/strided_slice_1/stack_2Þ
7Double_Bidirectional_GRU/forward_rnn_25/strided_slice_1StridedSlice8Double_Bidirectional_GRU/forward_rnn_25/Shape_1:output:0FDouble_Bidirectional_GRU/forward_rnn_25/strided_slice_1/stack:output:0HDouble_Bidirectional_GRU/forward_rnn_25/strided_slice_1/stack_1:output:0HDouble_Bidirectional_GRU/forward_rnn_25/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask29
7Double_Bidirectional_GRU/forward_rnn_25/strided_slice_1Õ
CDouble_Bidirectional_GRU/forward_rnn_25/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2E
CDouble_Bidirectional_GRU/forward_rnn_25/TensorArrayV2/element_shapeÒ
5Double_Bidirectional_GRU/forward_rnn_25/TensorArrayV2TensorListReserveLDouble_Bidirectional_GRU/forward_rnn_25/TensorArrayV2/element_shape:output:0@Double_Bidirectional_GRU/forward_rnn_25/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type027
5Double_Bidirectional_GRU/forward_rnn_25/TensorArrayV2
]Double_Bidirectional_GRU/forward_rnn_25/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2_
]Double_Bidirectional_GRU/forward_rnn_25/TensorArrayUnstack/TensorListFromTensor/element_shape
ODouble_Bidirectional_GRU/forward_rnn_25/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor5Double_Bidirectional_GRU/forward_rnn_25/transpose:y:0fDouble_Bidirectional_GRU/forward_rnn_25/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02Q
ODouble_Bidirectional_GRU/forward_rnn_25/TensorArrayUnstack/TensorListFromTensorÈ
=Double_Bidirectional_GRU/forward_rnn_25/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=Double_Bidirectional_GRU/forward_rnn_25/strided_slice_2/stackÌ
?Double_Bidirectional_GRU/forward_rnn_25/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2A
?Double_Bidirectional_GRU/forward_rnn_25/strided_slice_2/stack_1Ì
?Double_Bidirectional_GRU/forward_rnn_25/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?Double_Bidirectional_GRU/forward_rnn_25/strided_slice_2/stack_2í
7Double_Bidirectional_GRU/forward_rnn_25/strided_slice_2StridedSlice5Double_Bidirectional_GRU/forward_rnn_25/transpose:y:0FDouble_Bidirectional_GRU/forward_rnn_25/strided_slice_2/stack:output:0HDouble_Bidirectional_GRU/forward_rnn_25/strided_slice_2/stack_1:output:0HDouble_Bidirectional_GRU/forward_rnn_25/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask29
7Double_Bidirectional_GRU/forward_rnn_25/strided_slice_2
BDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/ReadVariableOpReadVariableOpKdouble_bidirectional_gru_forward_rnn_25_gru_cell_25_readvariableop_resource*
_output_shapes
:	*
dtype02D
BDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/ReadVariableOp
;Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/unstackUnpackJDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2=
;Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/unstack«
IDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMul/ReadVariableOpReadVariableOpRdouble_bidirectional_gru_forward_rnn_25_gru_cell_25_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02K
IDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMul/ReadVariableOpÊ
:Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMulMatMul@Double_Bidirectional_GRU/forward_rnn_25/strided_slice_2:output:0QDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2<
:Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMulÄ
;Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/BiasAddBiasAddDDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMul:product:0DDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2=
;Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/BiasAddÕ
CDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2E
CDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/split/split_dimÿ
9Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/splitSplitLDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/split/split_dim:output:0DDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2;
9Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/split±
KDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOpReadVariableOpTdouble_bidirectional_gru_forward_rnn_25_gru_cell_25_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02M
KDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOpÆ
<Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMul_1MatMul6Double_Bidirectional_GRU/forward_rnn_25/zeros:output:0SDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2>
<Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMul_1Ê
=Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/BiasAdd_1BiasAddFDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMul_1:product:0DDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2?
=Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/BiasAdd_1Ë
9Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2;
9Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/ConstÙ
EDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2G
EDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/split_1/split_dimØ
;Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/split_1SplitVFDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/BiasAdd_1:output:0BDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/Const:output:0NDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2=
;Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/split_1¸
7Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/addAddV2BDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/split:output:0DDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ29
7Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/addõ
;Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/SigmoidSigmoid;Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2=
;Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/Sigmoid¼
9Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/add_1AddV2BDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/split:output:1DDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2;
9Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/add_1û
=Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/Sigmoid_1Sigmoid=Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2?
=Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/Sigmoid_1µ
7Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/mulMulADouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/Sigmoid_1:y:0DDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ29
7Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/mul³
9Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/add_2AddV2BDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/split:output:2;Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2;
9Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/add_2î
8Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/TanhTanh=Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:
8Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/Tanh©
9Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/mul_1Mul?Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/Sigmoid:y:06Double_Bidirectional_GRU/forward_rnn_25/zeros:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2;
9Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/mul_1»
9Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2;
9Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/sub/x±
7Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/subSubBDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/sub/x:output:0?Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ29
7Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/sub«
9Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/mul_2Mul;Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/sub:z:0<Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2;
9Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/mul_2°
9Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/add_3AddV2=Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/mul_1:z:0=Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2;
9Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/add_3ß
EDouble_Bidirectional_GRU/forward_rnn_25/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2G
EDouble_Bidirectional_GRU/forward_rnn_25/TensorArrayV2_1/element_shapeØ
7Double_Bidirectional_GRU/forward_rnn_25/TensorArrayV2_1TensorListReserveNDouble_Bidirectional_GRU/forward_rnn_25/TensorArrayV2_1/element_shape:output:0@Double_Bidirectional_GRU/forward_rnn_25/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7Double_Bidirectional_GRU/forward_rnn_25/TensorArrayV2_1
,Double_Bidirectional_GRU/forward_rnn_25/timeConst*
_output_shapes
: *
dtype0*
value	B : 2.
,Double_Bidirectional_GRU/forward_rnn_25/timeÏ
@Double_Bidirectional_GRU/forward_rnn_25/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2B
@Double_Bidirectional_GRU/forward_rnn_25/while/maximum_iterationsº
:Double_Bidirectional_GRU/forward_rnn_25/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2<
:Double_Bidirectional_GRU/forward_rnn_25/while/loop_counterâ	
-Double_Bidirectional_GRU/forward_rnn_25/whileWhileCDouble_Bidirectional_GRU/forward_rnn_25/while/loop_counter:output:0IDouble_Bidirectional_GRU/forward_rnn_25/while/maximum_iterations:output:05Double_Bidirectional_GRU/forward_rnn_25/time:output:0@Double_Bidirectional_GRU/forward_rnn_25/TensorArrayV2_1:handle:06Double_Bidirectional_GRU/forward_rnn_25/zeros:output:0@Double_Bidirectional_GRU/forward_rnn_25/strided_slice_1:output:0_Double_Bidirectional_GRU/forward_rnn_25/TensorArrayUnstack/TensorListFromTensor:output_handle:0Kdouble_bidirectional_gru_forward_rnn_25_gru_cell_25_readvariableop_resourceRdouble_bidirectional_gru_forward_rnn_25_gru_cell_25_matmul_readvariableop_resourceTdouble_bidirectional_gru_forward_rnn_25_gru_cell_25_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*H
body@R>
<Double_Bidirectional_GRU_forward_rnn_25_while_body_387380744*H
cond@R>
<Double_Bidirectional_GRU_forward_rnn_25_while_cond_387380743*9
output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2/
-Double_Bidirectional_GRU/forward_rnn_25/while
XDouble_Bidirectional_GRU/forward_rnn_25/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2Z
XDouble_Bidirectional_GRU/forward_rnn_25/TensorArrayV2Stack/TensorListStack/element_shape
JDouble_Bidirectional_GRU/forward_rnn_25/TensorArrayV2Stack/TensorListStackTensorListStack6Double_Bidirectional_GRU/forward_rnn_25/while:output:3aDouble_Bidirectional_GRU/forward_rnn_25/TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02L
JDouble_Bidirectional_GRU/forward_rnn_25/TensorArrayV2Stack/TensorListStackÑ
=Double_Bidirectional_GRU/forward_rnn_25/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2?
=Double_Bidirectional_GRU/forward_rnn_25/strided_slice_3/stackÌ
?Double_Bidirectional_GRU/forward_rnn_25/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2A
?Double_Bidirectional_GRU/forward_rnn_25/strided_slice_3/stack_1Ì
?Double_Bidirectional_GRU/forward_rnn_25/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?Double_Bidirectional_GRU/forward_rnn_25/strided_slice_3/stack_2
7Double_Bidirectional_GRU/forward_rnn_25/strided_slice_3StridedSliceSDouble_Bidirectional_GRU/forward_rnn_25/TensorArrayV2Stack/TensorListStack:tensor:0FDouble_Bidirectional_GRU/forward_rnn_25/strided_slice_3/stack:output:0HDouble_Bidirectional_GRU/forward_rnn_25/strided_slice_3/stack_1:output:0HDouble_Bidirectional_GRU/forward_rnn_25/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask29
7Double_Bidirectional_GRU/forward_rnn_25/strided_slice_3É
8Double_Bidirectional_GRU/forward_rnn_25/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2:
8Double_Bidirectional_GRU/forward_rnn_25/transpose_1/permÏ
3Double_Bidirectional_GRU/forward_rnn_25/transpose_1	TransposeSDouble_Bidirectional_GRU/forward_rnn_25/TensorArrayV2Stack/TensorListStack:tensor:0ADouble_Bidirectional_GRU/forward_rnn_25/transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ25
3Double_Bidirectional_GRU/forward_rnn_25/transpose_1³
.Double_Bidirectional_GRU/backward_rnn_25/ShapeShape#Embedding_Dropout/Identity:output:0*
T0*
_output_shapes
:20
.Double_Bidirectional_GRU/backward_rnn_25/ShapeÆ
<Double_Bidirectional_GRU/backward_rnn_25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<Double_Bidirectional_GRU/backward_rnn_25/strided_slice/stackÊ
>Double_Bidirectional_GRU/backward_rnn_25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2@
>Double_Bidirectional_GRU/backward_rnn_25/strided_slice/stack_1Ê
>Double_Bidirectional_GRU/backward_rnn_25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>Double_Bidirectional_GRU/backward_rnn_25/strided_slice/stack_2Ø
6Double_Bidirectional_GRU/backward_rnn_25/strided_sliceStridedSlice7Double_Bidirectional_GRU/backward_rnn_25/Shape:output:0EDouble_Bidirectional_GRU/backward_rnn_25/strided_slice/stack:output:0GDouble_Bidirectional_GRU/backward_rnn_25/strided_slice/stack_1:output:0GDouble_Bidirectional_GRU/backward_rnn_25/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask28
6Double_Bidirectional_GRU/backward_rnn_25/strided_slice¯
4Double_Bidirectional_GRU/backward_rnn_25/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :26
4Double_Bidirectional_GRU/backward_rnn_25/zeros/mul/y
2Double_Bidirectional_GRU/backward_rnn_25/zeros/mulMul?Double_Bidirectional_GRU/backward_rnn_25/strided_slice:output:0=Double_Bidirectional_GRU/backward_rnn_25/zeros/mul/y:output:0*
T0*
_output_shapes
: 24
2Double_Bidirectional_GRU/backward_rnn_25/zeros/mul±
5Double_Bidirectional_GRU/backward_rnn_25/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è27
5Double_Bidirectional_GRU/backward_rnn_25/zeros/Less/y
3Double_Bidirectional_GRU/backward_rnn_25/zeros/LessLess6Double_Bidirectional_GRU/backward_rnn_25/zeros/mul:z:0>Double_Bidirectional_GRU/backward_rnn_25/zeros/Less/y:output:0*
T0*
_output_shapes
: 25
3Double_Bidirectional_GRU/backward_rnn_25/zeros/Lessµ
7Double_Bidirectional_GRU/backward_rnn_25/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :29
7Double_Bidirectional_GRU/backward_rnn_25/zeros/packed/1§
5Double_Bidirectional_GRU/backward_rnn_25/zeros/packedPack?Double_Bidirectional_GRU/backward_rnn_25/strided_slice:output:0@Double_Bidirectional_GRU/backward_rnn_25/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:27
5Double_Bidirectional_GRU/backward_rnn_25/zeros/packed±
4Double_Bidirectional_GRU/backward_rnn_25/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    26
4Double_Bidirectional_GRU/backward_rnn_25/zeros/Const
.Double_Bidirectional_GRU/backward_rnn_25/zerosFill>Double_Bidirectional_GRU/backward_rnn_25/zeros/packed:output:0=Double_Bidirectional_GRU/backward_rnn_25/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.Double_Bidirectional_GRU/backward_rnn_25/zerosÇ
7Double_Bidirectional_GRU/backward_rnn_25/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          29
7Double_Bidirectional_GRU/backward_rnn_25/transpose/perm
2Double_Bidirectional_GRU/backward_rnn_25/transpose	Transpose#Embedding_Dropout/Identity:output:0@Double_Bidirectional_GRU/backward_rnn_25/transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ24
2Double_Bidirectional_GRU/backward_rnn_25/transposeÊ
0Double_Bidirectional_GRU/backward_rnn_25/Shape_1Shape6Double_Bidirectional_GRU/backward_rnn_25/transpose:y:0*
T0*
_output_shapes
:22
0Double_Bidirectional_GRU/backward_rnn_25/Shape_1Ê
>Double_Bidirectional_GRU/backward_rnn_25/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2@
>Double_Bidirectional_GRU/backward_rnn_25/strided_slice_1/stackÎ
@Double_Bidirectional_GRU/backward_rnn_25/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2B
@Double_Bidirectional_GRU/backward_rnn_25/strided_slice_1/stack_1Î
@Double_Bidirectional_GRU/backward_rnn_25/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@Double_Bidirectional_GRU/backward_rnn_25/strided_slice_1/stack_2ä
8Double_Bidirectional_GRU/backward_rnn_25/strided_slice_1StridedSlice9Double_Bidirectional_GRU/backward_rnn_25/Shape_1:output:0GDouble_Bidirectional_GRU/backward_rnn_25/strided_slice_1/stack:output:0IDouble_Bidirectional_GRU/backward_rnn_25/strided_slice_1/stack_1:output:0IDouble_Bidirectional_GRU/backward_rnn_25/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2:
8Double_Bidirectional_GRU/backward_rnn_25/strided_slice_1×
DDouble_Bidirectional_GRU/backward_rnn_25/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2F
DDouble_Bidirectional_GRU/backward_rnn_25/TensorArrayV2/element_shapeÖ
6Double_Bidirectional_GRU/backward_rnn_25/TensorArrayV2TensorListReserveMDouble_Bidirectional_GRU/backward_rnn_25/TensorArrayV2/element_shape:output:0ADouble_Bidirectional_GRU/backward_rnn_25/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type028
6Double_Bidirectional_GRU/backward_rnn_25/TensorArrayV2¼
7Double_Bidirectional_GRU/backward_rnn_25/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 29
7Double_Bidirectional_GRU/backward_rnn_25/ReverseV2/axis¯
2Double_Bidirectional_GRU/backward_rnn_25/ReverseV2	ReverseV26Double_Bidirectional_GRU/backward_rnn_25/transpose:y:0@Double_Bidirectional_GRU/backward_rnn_25/ReverseV2/axis:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ24
2Double_Bidirectional_GRU/backward_rnn_25/ReverseV2
^Double_Bidirectional_GRU/backward_rnn_25/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2`
^Double_Bidirectional_GRU/backward_rnn_25/TensorArrayUnstack/TensorListFromTensor/element_shape¡
PDouble_Bidirectional_GRU/backward_rnn_25/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor;Double_Bidirectional_GRU/backward_rnn_25/ReverseV2:output:0gDouble_Bidirectional_GRU/backward_rnn_25/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02R
PDouble_Bidirectional_GRU/backward_rnn_25/TensorArrayUnstack/TensorListFromTensorÊ
>Double_Bidirectional_GRU/backward_rnn_25/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2@
>Double_Bidirectional_GRU/backward_rnn_25/strided_slice_2/stackÎ
@Double_Bidirectional_GRU/backward_rnn_25/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2B
@Double_Bidirectional_GRU/backward_rnn_25/strided_slice_2/stack_1Î
@Double_Bidirectional_GRU/backward_rnn_25/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@Double_Bidirectional_GRU/backward_rnn_25/strided_slice_2/stack_2ó
8Double_Bidirectional_GRU/backward_rnn_25/strided_slice_2StridedSlice6Double_Bidirectional_GRU/backward_rnn_25/transpose:y:0GDouble_Bidirectional_GRU/backward_rnn_25/strided_slice_2/stack:output:0IDouble_Bidirectional_GRU/backward_rnn_25/strided_slice_2/stack_1:output:0IDouble_Bidirectional_GRU/backward_rnn_25/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2:
8Double_Bidirectional_GRU/backward_rnn_25/strided_slice_2
CDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/ReadVariableOpReadVariableOpLdouble_bidirectional_gru_backward_rnn_25_gru_cell_25_readvariableop_resource*
_output_shapes
:	*
dtype02E
CDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/ReadVariableOp
<Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/unstackUnpackKDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2>
<Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/unstack®
JDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMul/ReadVariableOpReadVariableOpSdouble_bidirectional_gru_backward_rnn_25_gru_cell_25_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02L
JDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMul/ReadVariableOpÎ
;Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMulMatMulADouble_Bidirectional_GRU/backward_rnn_25/strided_slice_2:output:0RDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2=
;Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMulÈ
<Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/BiasAddBiasAddEDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMul:product:0EDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2>
<Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/BiasAdd×
DDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2F
DDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/split/split_dim
:Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/splitSplitMDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/split/split_dim:output:0EDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2<
:Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/split´
LDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOpReadVariableOpUdouble_bidirectional_gru_backward_rnn_25_gru_cell_25_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02N
LDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOpÊ
=Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMul_1MatMul7Double_Bidirectional_GRU/backward_rnn_25/zeros:output:0TDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2?
=Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMul_1Î
>Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/BiasAdd_1BiasAddGDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMul_1:product:0EDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2@
>Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/BiasAdd_1Í
:Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2<
:Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/ConstÛ
FDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2H
FDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/split_1/split_dimÝ
<Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/split_1SplitVGDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/BiasAdd_1:output:0CDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/Const:output:0ODouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2>
<Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/split_1¼
8Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/addAddV2CDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/split:output:0EDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:
8Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/addø
<Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/SigmoidSigmoid<Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2>
<Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/SigmoidÀ
:Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/add_1AddV2CDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/split:output:1EDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2<
:Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/add_1þ
>Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/Sigmoid_1Sigmoid>Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2@
>Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/Sigmoid_1¹
8Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/mulMulBDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/Sigmoid_1:y:0EDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:
8Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/mul·
:Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/add_2AddV2CDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/split:output:2<Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2<
:Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/add_2ñ
9Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/TanhTanh>Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2;
9Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/Tanh­
:Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/mul_1Mul@Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/Sigmoid:y:07Double_Bidirectional_GRU/backward_rnn_25/zeros:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2<
:Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/mul_1½
:Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2<
:Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/sub/xµ
8Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/subSubCDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/sub/x:output:0@Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:
8Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/sub¯
:Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/mul_2Mul<Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/sub:z:0=Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2<
:Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/mul_2´
:Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/add_3AddV2>Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/mul_1:z:0>Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2<
:Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/add_3á
FDouble_Bidirectional_GRU/backward_rnn_25/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2H
FDouble_Bidirectional_GRU/backward_rnn_25/TensorArrayV2_1/element_shapeÜ
8Double_Bidirectional_GRU/backward_rnn_25/TensorArrayV2_1TensorListReserveODouble_Bidirectional_GRU/backward_rnn_25/TensorArrayV2_1/element_shape:output:0ADouble_Bidirectional_GRU/backward_rnn_25/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8Double_Bidirectional_GRU/backward_rnn_25/TensorArrayV2_1 
-Double_Bidirectional_GRU/backward_rnn_25/timeConst*
_output_shapes
: *
dtype0*
value	B : 2/
-Double_Bidirectional_GRU/backward_rnn_25/timeÑ
ADouble_Bidirectional_GRU/backward_rnn_25/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2C
ADouble_Bidirectional_GRU/backward_rnn_25/while/maximum_iterations¼
;Double_Bidirectional_GRU/backward_rnn_25/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2=
;Double_Bidirectional_GRU/backward_rnn_25/while/loop_counterð	
.Double_Bidirectional_GRU/backward_rnn_25/whileWhileDDouble_Bidirectional_GRU/backward_rnn_25/while/loop_counter:output:0JDouble_Bidirectional_GRU/backward_rnn_25/while/maximum_iterations:output:06Double_Bidirectional_GRU/backward_rnn_25/time:output:0ADouble_Bidirectional_GRU/backward_rnn_25/TensorArrayV2_1:handle:07Double_Bidirectional_GRU/backward_rnn_25/zeros:output:0ADouble_Bidirectional_GRU/backward_rnn_25/strided_slice_1:output:0`Double_Bidirectional_GRU/backward_rnn_25/TensorArrayUnstack/TensorListFromTensor:output_handle:0Ldouble_bidirectional_gru_backward_rnn_25_gru_cell_25_readvariableop_resourceSdouble_bidirectional_gru_backward_rnn_25_gru_cell_25_matmul_readvariableop_resourceUdouble_bidirectional_gru_backward_rnn_25_gru_cell_25_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*I
bodyAR?
=Double_Bidirectional_GRU_backward_rnn_25_while_body_387380898*I
condAR?
=Double_Bidirectional_GRU_backward_rnn_25_while_cond_387380897*9
output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 20
.Double_Bidirectional_GRU/backward_rnn_25/while
YDouble_Bidirectional_GRU/backward_rnn_25/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2[
YDouble_Bidirectional_GRU/backward_rnn_25/TensorArrayV2Stack/TensorListStack/element_shape
KDouble_Bidirectional_GRU/backward_rnn_25/TensorArrayV2Stack/TensorListStackTensorListStack7Double_Bidirectional_GRU/backward_rnn_25/while:output:3bDouble_Bidirectional_GRU/backward_rnn_25/TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02M
KDouble_Bidirectional_GRU/backward_rnn_25/TensorArrayV2Stack/TensorListStackÓ
>Double_Bidirectional_GRU/backward_rnn_25/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2@
>Double_Bidirectional_GRU/backward_rnn_25/strided_slice_3/stackÎ
@Double_Bidirectional_GRU/backward_rnn_25/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2B
@Double_Bidirectional_GRU/backward_rnn_25/strided_slice_3/stack_1Î
@Double_Bidirectional_GRU/backward_rnn_25/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@Double_Bidirectional_GRU/backward_rnn_25/strided_slice_3/stack_2
8Double_Bidirectional_GRU/backward_rnn_25/strided_slice_3StridedSliceTDouble_Bidirectional_GRU/backward_rnn_25/TensorArrayV2Stack/TensorListStack:tensor:0GDouble_Bidirectional_GRU/backward_rnn_25/strided_slice_3/stack:output:0IDouble_Bidirectional_GRU/backward_rnn_25/strided_slice_3/stack_1:output:0IDouble_Bidirectional_GRU/backward_rnn_25/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2:
8Double_Bidirectional_GRU/backward_rnn_25/strided_slice_3Ë
9Double_Bidirectional_GRU/backward_rnn_25/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2;
9Double_Bidirectional_GRU/backward_rnn_25/transpose_1/permÓ
4Double_Bidirectional_GRU/backward_rnn_25/transpose_1	TransposeTDouble_Bidirectional_GRU/backward_rnn_25/TensorArrayV2Stack/TensorListStack:tensor:0BDouble_Bidirectional_GRU/backward_rnn_25/transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ26
4Double_Bidirectional_GRU/backward_rnn_25/transpose_1
$Double_Bidirectional_GRU/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2&
$Double_Bidirectional_GRU/concat/axis¾
Double_Bidirectional_GRU/concatConcatV2@Double_Bidirectional_GRU/forward_rnn_25/strided_slice_3:output:0ADouble_Bidirectional_GRU/backward_rnn_25/strided_slice_3:output:0-Double_Bidirectional_GRU/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
Double_Bidirectional_GRU/concat¯
 Prediction/MatMul/ReadVariableOpReadVariableOp)prediction_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 Prediction/MatMul/ReadVariableOp¶
Prediction/MatMulMatMul(Double_Bidirectional_GRU/concat:output:0(Prediction/MatMul/ReadVariableOp:value:0*
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
Prediction/SigmoidÓ
IdentityIdentityPrediction/Sigmoid:y:0K^Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMul/ReadVariableOpM^Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOpD^Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/ReadVariableOp/^Double_Bidirectional_GRU/backward_rnn_25/whileJ^Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMul/ReadVariableOpL^Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOpC^Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/ReadVariableOp.^Double_Bidirectional_GRU/forward_rnn_25/while"^Prediction/BiasAdd/ReadVariableOp!^Prediction/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : 2
JDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMul/ReadVariableOpJDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMul/ReadVariableOp2
LDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOpLDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp2
CDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/ReadVariableOpCDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/ReadVariableOp2`
.Double_Bidirectional_GRU/backward_rnn_25/while.Double_Bidirectional_GRU/backward_rnn_25/while2
IDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMul/ReadVariableOpIDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMul/ReadVariableOp2
KDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOpKDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp2
BDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/ReadVariableOpBDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/ReadVariableOp2^
-Double_Bidirectional_GRU/forward_rnn_25/while-Double_Bidirectional_GRU/forward_rnn_25/while2F
!Prediction/BiasAdd/ReadVariableOp!Prediction/BiasAdd/ReadVariableOp2D
 Prediction/MatMul/ReadVariableOp Prediction/MatMul/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ
n
P__inference_Embedding_Dropout_layer_call_and_return_conditional_losses_387379733

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
Ý
é
$backward_rnn_25_while_cond_387381916<
8backward_rnn_25_while_backward_rnn_25_while_loop_counterB
>backward_rnn_25_while_backward_rnn_25_while_maximum_iterations%
!backward_rnn_25_while_placeholder'
#backward_rnn_25_while_placeholder_1'
#backward_rnn_25_while_placeholder_2>
:backward_rnn_25_while_less_backward_rnn_25_strided_slice_1W
Sbackward_rnn_25_while_backward_rnn_25_while_cond_387381916___redundant_placeholder0W
Sbackward_rnn_25_while_backward_rnn_25_while_cond_387381916___redundant_placeholder1W
Sbackward_rnn_25_while_backward_rnn_25_while_cond_387381916___redundant_placeholder2W
Sbackward_rnn_25_while_backward_rnn_25_while_cond_387381916___redundant_placeholder3"
backward_rnn_25_while_identity
À
backward_rnn_25/while/LessLess!backward_rnn_25_while_placeholder:backward_rnn_25_while_less_backward_rnn_25_strided_slice_1*
T0*
_output_shapes
: 2
backward_rnn_25/while/Less
backward_rnn_25/while/IdentityIdentitybackward_rnn_25/while/Less:z:0*
T0
*
_output_shapes
: 2 
backward_rnn_25/while/Identity"I
backward_rnn_25_while_identity'backward_rnn_25/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:

n
5__inference_Embedding_Dropout_layer_call_fn_387381332

inputs
identity¢StatefulPartitionedCall÷
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
GPU2*0J 8 *Y
fTRR
P__inference_Embedding_Dropout_layer_call_and_return_conditional_losses_3873804752
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

¹	
<Double_Bidirectional_GRU_forward_rnn_25_while_cond_387381070l
hdouble_bidirectional_gru_forward_rnn_25_while_double_bidirectional_gru_forward_rnn_25_while_loop_counterr
ndouble_bidirectional_gru_forward_rnn_25_while_double_bidirectional_gru_forward_rnn_25_while_maximum_iterations=
9double_bidirectional_gru_forward_rnn_25_while_placeholder?
;double_bidirectional_gru_forward_rnn_25_while_placeholder_1?
;double_bidirectional_gru_forward_rnn_25_while_placeholder_2n
jdouble_bidirectional_gru_forward_rnn_25_while_less_double_bidirectional_gru_forward_rnn_25_strided_slice_1
double_bidirectional_gru_forward_rnn_25_while_double_bidirectional_gru_forward_rnn_25_while_cond_387381070___redundant_placeholder0
double_bidirectional_gru_forward_rnn_25_while_double_bidirectional_gru_forward_rnn_25_while_cond_387381070___redundant_placeholder1
double_bidirectional_gru_forward_rnn_25_while_double_bidirectional_gru_forward_rnn_25_while_cond_387381070___redundant_placeholder2
double_bidirectional_gru_forward_rnn_25_while_double_bidirectional_gru_forward_rnn_25_while_cond_387381070___redundant_placeholder3:
6double_bidirectional_gru_forward_rnn_25_while_identity
¸
2Double_Bidirectional_GRU/forward_rnn_25/while/LessLess9double_bidirectional_gru_forward_rnn_25_while_placeholderjdouble_bidirectional_gru_forward_rnn_25_while_less_double_bidirectional_gru_forward_rnn_25_strided_slice_1*
T0*
_output_shapes
: 24
2Double_Bidirectional_GRU/forward_rnn_25/while/LessÕ
6Double_Bidirectional_GRU/forward_rnn_25/while/IdentityIdentity6Double_Bidirectional_GRU/forward_rnn_25/while/Less:z:0*
T0
*
_output_shapes
: 28
6Double_Bidirectional_GRU/forward_rnn_25/while/Identity"y
6double_bidirectional_gru_forward_rnn_25_while_identity?Double_Bidirectional_GRU/forward_rnn_25/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:

¹
while_cond_387379421
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_17
3while_while_cond_387379421___redundant_placeholder07
3while_while_cond_387379421___redundant_placeholder17
3while_while_cond_387379421___redundant_placeholder27
3while_while_cond_387379421___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ß
à
J__inference_gru_cell_25_layer_call_and_return_conditional_losses_387382805

inputs
states_0*
readvariableop_resource:	2
matmul_readvariableop_resource:
4
 matmul_1_readvariableop_resource:

identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpy
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
split/split_dim¯
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	BiasAdd_1c
ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2
Constq
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
split_1/split_dimÔ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanh_
mul_1MulSigmoid:y:0states_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sub[
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_3¡
IdentityIdentity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¥

Identity_1Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0
º
Ì	
=Double_Bidirectional_GRU_backward_rnn_25_while_cond_387381224n
jdouble_bidirectional_gru_backward_rnn_25_while_double_bidirectional_gru_backward_rnn_25_while_loop_countert
pdouble_bidirectional_gru_backward_rnn_25_while_double_bidirectional_gru_backward_rnn_25_while_maximum_iterations>
:double_bidirectional_gru_backward_rnn_25_while_placeholder@
<double_bidirectional_gru_backward_rnn_25_while_placeholder_1@
<double_bidirectional_gru_backward_rnn_25_while_placeholder_2p
ldouble_bidirectional_gru_backward_rnn_25_while_less_double_bidirectional_gru_backward_rnn_25_strided_slice_1
double_bidirectional_gru_backward_rnn_25_while_double_bidirectional_gru_backward_rnn_25_while_cond_387381224___redundant_placeholder0
double_bidirectional_gru_backward_rnn_25_while_double_bidirectional_gru_backward_rnn_25_while_cond_387381224___redundant_placeholder1
double_bidirectional_gru_backward_rnn_25_while_double_bidirectional_gru_backward_rnn_25_while_cond_387381224___redundant_placeholder2
double_bidirectional_gru_backward_rnn_25_while_double_bidirectional_gru_backward_rnn_25_while_cond_387381224___redundant_placeholder3;
7double_bidirectional_gru_backward_rnn_25_while_identity
½
3Double_Bidirectional_GRU/backward_rnn_25/while/LessLess:double_bidirectional_gru_backward_rnn_25_while_placeholderldouble_bidirectional_gru_backward_rnn_25_while_less_double_bidirectional_gru_backward_rnn_25_strided_slice_1*
T0*
_output_shapes
: 25
3Double_Bidirectional_GRU/backward_rnn_25/while/LessØ
7Double_Bidirectional_GRU/backward_rnn_25/while/IdentityIdentity7Double_Bidirectional_GRU/backward_rnn_25/while/Less:z:0*
T0
*
_output_shapes
: 29
7Double_Bidirectional_GRU/backward_rnn_25/while/Identity"{
7double_bidirectional_gru_backward_rnn_25_while_identity@Double_Bidirectional_GRU/backward_rnn_25/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
H

while_body_387382117
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_25_readvariableop_resource_0:	F
2while_gru_cell_25_matmul_readvariableop_resource_0:
H
4while_gru_cell_25_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_25_readvariableop_resource:	D
0while_gru_cell_25_matmul_readvariableop_resource:
F
2while_gru_cell_25_matmul_1_readvariableop_resource:
¢'while/gru_cell_25/MatMul/ReadVariableOp¢)while/gru_cell_25/MatMul_1/ReadVariableOp¢ while/gru_cell_25/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem±
 while/gru_cell_25/ReadVariableOpReadVariableOp+while_gru_cell_25_readvariableop_resource_0*
_output_shapes
:	*
dtype02"
 while/gru_cell_25/ReadVariableOp¢
while/gru_cell_25/unstackUnpack(while/gru_cell_25/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
while/gru_cell_25/unstackÇ
'while/gru_cell_25/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_25_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype02)
'while/gru_cell_25/MatMul/ReadVariableOpÔ
while/gru_cell_25/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/MatMul¼
while/gru_cell_25/BiasAddBiasAdd"while/gru_cell_25/MatMul:product:0"while/gru_cell_25/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/BiasAdd
!while/gru_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2#
!while/gru_cell_25/split/split_dim÷
while/gru_cell_25/splitSplit*while/gru_cell_25/split/split_dim:output:0"while/gru_cell_25/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/gru_cell_25/splitÍ
)while/gru_cell_25/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_25_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02+
)while/gru_cell_25/MatMul_1/ReadVariableOp½
while/gru_cell_25/MatMul_1MatMulwhile_placeholder_21while/gru_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/MatMul_1Â
while/gru_cell_25/BiasAdd_1BiasAdd$while/gru_cell_25/MatMul_1:product:0"while/gru_cell_25/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/BiasAdd_1
while/gru_cell_25/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2
while/gru_cell_25/Const
#while/gru_cell_25/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#while/gru_cell_25/split_1/split_dim®
while/gru_cell_25/split_1SplitV$while/gru_cell_25/BiasAdd_1:output:0 while/gru_cell_25/Const:output:0,while/gru_cell_25/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/gru_cell_25/split_1°
while/gru_cell_25/addAddV2 while/gru_cell_25/split:output:0"while/gru_cell_25/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/add
while/gru_cell_25/SigmoidSigmoidwhile/gru_cell_25/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/Sigmoid´
while/gru_cell_25/add_1AddV2 while/gru_cell_25/split:output:1"while/gru_cell_25/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/add_1
while/gru_cell_25/Sigmoid_1Sigmoidwhile/gru_cell_25/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/Sigmoid_1­
while/gru_cell_25/mulMulwhile/gru_cell_25/Sigmoid_1:y:0"while/gru_cell_25/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/mul«
while/gru_cell_25/add_2AddV2 while/gru_cell_25/split:output:2while/gru_cell_25/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/add_2
while/gru_cell_25/TanhTanhwhile/gru_cell_25/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/Tanh 
while/gru_cell_25/mul_1Mulwhile/gru_cell_25/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/mul_1w
while/gru_cell_25/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_25/sub/x©
while/gru_cell_25/subSub while/gru_cell_25/sub/x:output:0while/gru_cell_25/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/sub£
while/gru_cell_25/mul_2Mulwhile/gru_cell_25/sub:z:0while/gru_cell_25/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/mul_2¨
while/gru_cell_25/add_3AddV2while/gru_cell_25/mul_1:z:0while/gru_cell_25/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/add_3ß
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_25/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1×
while/IdentityIdentitywhile/add_1:z:0(^while/gru_cell_25/MatMul/ReadVariableOp*^while/gru_cell_25/MatMul_1/ReadVariableOp!^while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityê
while/Identity_1Identitywhile_while_maximum_iterations(^while/gru_cell_25/MatMul/ReadVariableOp*^while/gru_cell_25/MatMul_1/ReadVariableOp!^while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1Ù
while/Identity_2Identitywhile/add:z:0(^while/gru_cell_25/MatMul/ReadVariableOp*^while/gru_cell_25/MatMul_1/ReadVariableOp!^while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/gru_cell_25/MatMul/ReadVariableOp*^while/gru_cell_25/MatMul_1/ReadVariableOp!^while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3ù
while/Identity_4Identitywhile/gru_cell_25/add_3:z:0(^while/gru_cell_25/MatMul/ReadVariableOp*^while/gru_cell_25/MatMul_1/ReadVariableOp!^while/gru_cell_25/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4"j
2while_gru_cell_25_matmul_1_readvariableop_resource4while_gru_cell_25_matmul_1_readvariableop_resource_0"f
0while_gru_cell_25_matmul_readvariableop_resource2while_gru_cell_25_matmul_readvariableop_resource_0"X
)while_gru_cell_25_readvariableop_resource+while_gru_cell_25_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : 2R
'while/gru_cell_25/MatMul/ReadVariableOp'while/gru_cell_25/MatMul/ReadVariableOp2V
)while/gru_cell_25/MatMul_1/ReadVariableOp)while/gru_cell_25/MatMul_1/ReadVariableOp2D
 while/gru_cell_25/ReadVariableOp while/gru_cell_25/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ñ
¡
W__inference_Double_Bidirectional_GRU_layer_call_and_return_conditional_losses_387382007

inputsE
2forward_rnn_25_gru_cell_25_readvariableop_resource:	M
9forward_rnn_25_gru_cell_25_matmul_readvariableop_resource:
O
;forward_rnn_25_gru_cell_25_matmul_1_readvariableop_resource:
F
3backward_rnn_25_gru_cell_25_readvariableop_resource:	N
:backward_rnn_25_gru_cell_25_matmul_readvariableop_resource:
P
<backward_rnn_25_gru_cell_25_matmul_1_readvariableop_resource:

identity¢1backward_rnn_25/gru_cell_25/MatMul/ReadVariableOp¢3backward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp¢*backward_rnn_25/gru_cell_25/ReadVariableOp¢backward_rnn_25/while¢0forward_rnn_25/gru_cell_25/MatMul/ReadVariableOp¢2forward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp¢)forward_rnn_25/gru_cell_25/ReadVariableOp¢forward_rnn_25/whileb
forward_rnn_25/ShapeShapeinputs*
T0*
_output_shapes
:2
forward_rnn_25/Shape
"forward_rnn_25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"forward_rnn_25/strided_slice/stack
$forward_rnn_25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$forward_rnn_25/strided_slice/stack_1
$forward_rnn_25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$forward_rnn_25/strided_slice/stack_2¼
forward_rnn_25/strided_sliceStridedSliceforward_rnn_25/Shape:output:0+forward_rnn_25/strided_slice/stack:output:0-forward_rnn_25/strided_slice/stack_1:output:0-forward_rnn_25/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
forward_rnn_25/strided_slice{
forward_rnn_25/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
forward_rnn_25/zeros/mul/y¨
forward_rnn_25/zeros/mulMul%forward_rnn_25/strided_slice:output:0#forward_rnn_25/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_rnn_25/zeros/mul}
forward_rnn_25/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
forward_rnn_25/zeros/Less/y£
forward_rnn_25/zeros/LessLessforward_rnn_25/zeros/mul:z:0$forward_rnn_25/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_rnn_25/zeros/Less
forward_rnn_25/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
forward_rnn_25/zeros/packed/1¿
forward_rnn_25/zeros/packedPack%forward_rnn_25/strided_slice:output:0&forward_rnn_25/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_rnn_25/zeros/packed}
forward_rnn_25/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_rnn_25/zeros/Const²
forward_rnn_25/zerosFill$forward_rnn_25/zeros/packed:output:0#forward_rnn_25/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
forward_rnn_25/zeros
forward_rnn_25/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
forward_rnn_25/transpose/perm±
forward_rnn_25/transpose	Transposeinputs&forward_rnn_25/transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
forward_rnn_25/transpose|
forward_rnn_25/Shape_1Shapeforward_rnn_25/transpose:y:0*
T0*
_output_shapes
:2
forward_rnn_25/Shape_1
$forward_rnn_25/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_rnn_25/strided_slice_1/stack
&forward_rnn_25/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_rnn_25/strided_slice_1/stack_1
&forward_rnn_25/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_rnn_25/strided_slice_1/stack_2È
forward_rnn_25/strided_slice_1StridedSliceforward_rnn_25/Shape_1:output:0-forward_rnn_25/strided_slice_1/stack:output:0/forward_rnn_25/strided_slice_1/stack_1:output:0/forward_rnn_25/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
forward_rnn_25/strided_slice_1£
*forward_rnn_25/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2,
*forward_rnn_25/TensorArrayV2/element_shapeî
forward_rnn_25/TensorArrayV2TensorListReserve3forward_rnn_25/TensorArrayV2/element_shape:output:0'forward_rnn_25/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
forward_rnn_25/TensorArrayV2Ý
Dforward_rnn_25/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2F
Dforward_rnn_25/TensorArrayUnstack/TensorListFromTensor/element_shape´
6forward_rnn_25/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_rnn_25/transpose:y:0Mforward_rnn_25/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type028
6forward_rnn_25/TensorArrayUnstack/TensorListFromTensor
$forward_rnn_25/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_rnn_25/strided_slice_2/stack
&forward_rnn_25/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_rnn_25/strided_slice_2/stack_1
&forward_rnn_25/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_rnn_25/strided_slice_2/stack_2×
forward_rnn_25/strided_slice_2StridedSliceforward_rnn_25/transpose:y:0-forward_rnn_25/strided_slice_2/stack:output:0/forward_rnn_25/strided_slice_2/stack_1:output:0/forward_rnn_25/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2 
forward_rnn_25/strided_slice_2Ê
)forward_rnn_25/gru_cell_25/ReadVariableOpReadVariableOp2forward_rnn_25_gru_cell_25_readvariableop_resource*
_output_shapes
:	*
dtype02+
)forward_rnn_25/gru_cell_25/ReadVariableOp½
"forward_rnn_25/gru_cell_25/unstackUnpack1forward_rnn_25/gru_cell_25/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2$
"forward_rnn_25/gru_cell_25/unstackà
0forward_rnn_25/gru_cell_25/MatMul/ReadVariableOpReadVariableOp9forward_rnn_25_gru_cell_25_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype022
0forward_rnn_25/gru_cell_25/MatMul/ReadVariableOpæ
!forward_rnn_25/gru_cell_25/MatMulMatMul'forward_rnn_25/strided_slice_2:output:08forward_rnn_25/gru_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!forward_rnn_25/gru_cell_25/MatMulà
"forward_rnn_25/gru_cell_25/BiasAddBiasAdd+forward_rnn_25/gru_cell_25/MatMul:product:0+forward_rnn_25/gru_cell_25/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"forward_rnn_25/gru_cell_25/BiasAdd£
*forward_rnn_25/gru_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2,
*forward_rnn_25/gru_cell_25/split/split_dim
 forward_rnn_25/gru_cell_25/splitSplit3forward_rnn_25/gru_cell_25/split/split_dim:output:0+forward_rnn_25/gru_cell_25/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2"
 forward_rnn_25/gru_cell_25/splitæ
2forward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOpReadVariableOp;forward_rnn_25_gru_cell_25_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype024
2forward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOpâ
#forward_rnn_25/gru_cell_25/MatMul_1MatMulforward_rnn_25/zeros:output:0:forward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#forward_rnn_25/gru_cell_25/MatMul_1æ
$forward_rnn_25/gru_cell_25/BiasAdd_1BiasAdd-forward_rnn_25/gru_cell_25/MatMul_1:product:0+forward_rnn_25/gru_cell_25/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$forward_rnn_25/gru_cell_25/BiasAdd_1
 forward_rnn_25/gru_cell_25/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2"
 forward_rnn_25/gru_cell_25/Const§
,forward_rnn_25/gru_cell_25/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2.
,forward_rnn_25/gru_cell_25/split_1/split_dimÛ
"forward_rnn_25/gru_cell_25/split_1SplitV-forward_rnn_25/gru_cell_25/BiasAdd_1:output:0)forward_rnn_25/gru_cell_25/Const:output:05forward_rnn_25/gru_cell_25/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2$
"forward_rnn_25/gru_cell_25/split_1Ô
forward_rnn_25/gru_cell_25/addAddV2)forward_rnn_25/gru_cell_25/split:output:0+forward_rnn_25/gru_cell_25/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
forward_rnn_25/gru_cell_25/addª
"forward_rnn_25/gru_cell_25/SigmoidSigmoid"forward_rnn_25/gru_cell_25/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"forward_rnn_25/gru_cell_25/SigmoidØ
 forward_rnn_25/gru_cell_25/add_1AddV2)forward_rnn_25/gru_cell_25/split:output:1+forward_rnn_25/gru_cell_25/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 forward_rnn_25/gru_cell_25/add_1°
$forward_rnn_25/gru_cell_25/Sigmoid_1Sigmoid$forward_rnn_25/gru_cell_25/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$forward_rnn_25/gru_cell_25/Sigmoid_1Ñ
forward_rnn_25/gru_cell_25/mulMul(forward_rnn_25/gru_cell_25/Sigmoid_1:y:0+forward_rnn_25/gru_cell_25/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
forward_rnn_25/gru_cell_25/mulÏ
 forward_rnn_25/gru_cell_25/add_2AddV2)forward_rnn_25/gru_cell_25/split:output:2"forward_rnn_25/gru_cell_25/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 forward_rnn_25/gru_cell_25/add_2£
forward_rnn_25/gru_cell_25/TanhTanh$forward_rnn_25/gru_cell_25/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
forward_rnn_25/gru_cell_25/TanhÅ
 forward_rnn_25/gru_cell_25/mul_1Mul&forward_rnn_25/gru_cell_25/Sigmoid:y:0forward_rnn_25/zeros:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 forward_rnn_25/gru_cell_25/mul_1
 forward_rnn_25/gru_cell_25/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2"
 forward_rnn_25/gru_cell_25/sub/xÍ
forward_rnn_25/gru_cell_25/subSub)forward_rnn_25/gru_cell_25/sub/x:output:0&forward_rnn_25/gru_cell_25/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
forward_rnn_25/gru_cell_25/subÇ
 forward_rnn_25/gru_cell_25/mul_2Mul"forward_rnn_25/gru_cell_25/sub:z:0#forward_rnn_25/gru_cell_25/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 forward_rnn_25/gru_cell_25/mul_2Ì
 forward_rnn_25/gru_cell_25/add_3AddV2$forward_rnn_25/gru_cell_25/mul_1:z:0$forward_rnn_25/gru_cell_25/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 forward_rnn_25/gru_cell_25/add_3­
,forward_rnn_25/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2.
,forward_rnn_25/TensorArrayV2_1/element_shapeô
forward_rnn_25/TensorArrayV2_1TensorListReserve5forward_rnn_25/TensorArrayV2_1/element_shape:output:0'forward_rnn_25/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
forward_rnn_25/TensorArrayV2_1l
forward_rnn_25/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_rnn_25/time
'forward_rnn_25/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2)
'forward_rnn_25/while/maximum_iterations
!forward_rnn_25/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2#
!forward_rnn_25/while/loop_counter
forward_rnn_25/whileWhile*forward_rnn_25/while/loop_counter:output:00forward_rnn_25/while/maximum_iterations:output:0forward_rnn_25/time:output:0'forward_rnn_25/TensorArrayV2_1:handle:0forward_rnn_25/zeros:output:0'forward_rnn_25/strided_slice_1:output:0Fforward_rnn_25/TensorArrayUnstack/TensorListFromTensor:output_handle:02forward_rnn_25_gru_cell_25_readvariableop_resource9forward_rnn_25_gru_cell_25_matmul_readvariableop_resource;forward_rnn_25_gru_cell_25_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*/
body'R%
#forward_rnn_25_while_body_387381763*/
cond'R%
#forward_rnn_25_while_cond_387381762*9
output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
forward_rnn_25/whileÓ
?forward_rnn_25/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2A
?forward_rnn_25/TensorArrayV2Stack/TensorListStack/element_shape®
1forward_rnn_25/TensorArrayV2Stack/TensorListStackTensorListStackforward_rnn_25/while:output:3Hforward_rnn_25/TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype023
1forward_rnn_25/TensorArrayV2Stack/TensorListStack
$forward_rnn_25/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2&
$forward_rnn_25/strided_slice_3/stack
&forward_rnn_25/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_rnn_25/strided_slice_3/stack_1
&forward_rnn_25/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_rnn_25/strided_slice_3/stack_2õ
forward_rnn_25/strided_slice_3StridedSlice:forward_rnn_25/TensorArrayV2Stack/TensorListStack:tensor:0-forward_rnn_25/strided_slice_3/stack:output:0/forward_rnn_25/strided_slice_3/stack_1:output:0/forward_rnn_25/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2 
forward_rnn_25/strided_slice_3
forward_rnn_25/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
forward_rnn_25/transpose_1/permë
forward_rnn_25/transpose_1	Transpose:forward_rnn_25/TensorArrayV2Stack/TensorListStack:tensor:0(forward_rnn_25/transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
forward_rnn_25/transpose_1d
backward_rnn_25/ShapeShapeinputs*
T0*
_output_shapes
:2
backward_rnn_25/Shape
#backward_rnn_25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#backward_rnn_25/strided_slice/stack
%backward_rnn_25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%backward_rnn_25/strided_slice/stack_1
%backward_rnn_25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%backward_rnn_25/strided_slice/stack_2Â
backward_rnn_25/strided_sliceStridedSlicebackward_rnn_25/Shape:output:0,backward_rnn_25/strided_slice/stack:output:0.backward_rnn_25/strided_slice/stack_1:output:0.backward_rnn_25/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
backward_rnn_25/strided_slice}
backward_rnn_25/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
backward_rnn_25/zeros/mul/y¬
backward_rnn_25/zeros/mulMul&backward_rnn_25/strided_slice:output:0$backward_rnn_25/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_rnn_25/zeros/mul
backward_rnn_25/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
backward_rnn_25/zeros/Less/y§
backward_rnn_25/zeros/LessLessbackward_rnn_25/zeros/mul:z:0%backward_rnn_25/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_rnn_25/zeros/Less
backward_rnn_25/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2 
backward_rnn_25/zeros/packed/1Ã
backward_rnn_25/zeros/packedPack&backward_rnn_25/strided_slice:output:0'backward_rnn_25/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
backward_rnn_25/zeros/packed
backward_rnn_25/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_rnn_25/zeros/Const¶
backward_rnn_25/zerosFill%backward_rnn_25/zeros/packed:output:0$backward_rnn_25/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
backward_rnn_25/zeros
backward_rnn_25/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
backward_rnn_25/transpose/perm´
backward_rnn_25/transpose	Transposeinputs'backward_rnn_25/transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
backward_rnn_25/transpose
backward_rnn_25/Shape_1Shapebackward_rnn_25/transpose:y:0*
T0*
_output_shapes
:2
backward_rnn_25/Shape_1
%backward_rnn_25/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_rnn_25/strided_slice_1/stack
'backward_rnn_25/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_rnn_25/strided_slice_1/stack_1
'backward_rnn_25/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_rnn_25/strided_slice_1/stack_2Î
backward_rnn_25/strided_slice_1StridedSlice backward_rnn_25/Shape_1:output:0.backward_rnn_25/strided_slice_1/stack:output:00backward_rnn_25/strided_slice_1/stack_1:output:00backward_rnn_25/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
backward_rnn_25/strided_slice_1¥
+backward_rnn_25/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2-
+backward_rnn_25/TensorArrayV2/element_shapeò
backward_rnn_25/TensorArrayV2TensorListReserve4backward_rnn_25/TensorArrayV2/element_shape:output:0(backward_rnn_25/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
backward_rnn_25/TensorArrayV2
backward_rnn_25/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2 
backward_rnn_25/ReverseV2/axisË
backward_rnn_25/ReverseV2	ReverseV2backward_rnn_25/transpose:y:0'backward_rnn_25/ReverseV2/axis:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
backward_rnn_25/ReverseV2ß
Ebackward_rnn_25/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2G
Ebackward_rnn_25/TensorArrayUnstack/TensorListFromTensor/element_shape½
7backward_rnn_25/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"backward_rnn_25/ReverseV2:output:0Nbackward_rnn_25/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7backward_rnn_25/TensorArrayUnstack/TensorListFromTensor
%backward_rnn_25/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_rnn_25/strided_slice_2/stack
'backward_rnn_25/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_rnn_25/strided_slice_2/stack_1
'backward_rnn_25/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_rnn_25/strided_slice_2/stack_2Ý
backward_rnn_25/strided_slice_2StridedSlicebackward_rnn_25/transpose:y:0.backward_rnn_25/strided_slice_2/stack:output:00backward_rnn_25/strided_slice_2/stack_1:output:00backward_rnn_25/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2!
backward_rnn_25/strided_slice_2Í
*backward_rnn_25/gru_cell_25/ReadVariableOpReadVariableOp3backward_rnn_25_gru_cell_25_readvariableop_resource*
_output_shapes
:	*
dtype02,
*backward_rnn_25/gru_cell_25/ReadVariableOpÀ
#backward_rnn_25/gru_cell_25/unstackUnpack2backward_rnn_25/gru_cell_25/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2%
#backward_rnn_25/gru_cell_25/unstackã
1backward_rnn_25/gru_cell_25/MatMul/ReadVariableOpReadVariableOp:backward_rnn_25_gru_cell_25_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype023
1backward_rnn_25/gru_cell_25/MatMul/ReadVariableOpê
"backward_rnn_25/gru_cell_25/MatMulMatMul(backward_rnn_25/strided_slice_2:output:09backward_rnn_25/gru_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"backward_rnn_25/gru_cell_25/MatMulä
#backward_rnn_25/gru_cell_25/BiasAddBiasAdd,backward_rnn_25/gru_cell_25/MatMul:product:0,backward_rnn_25/gru_cell_25/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#backward_rnn_25/gru_cell_25/BiasAdd¥
+backward_rnn_25/gru_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2-
+backward_rnn_25/gru_cell_25/split/split_dim
!backward_rnn_25/gru_cell_25/splitSplit4backward_rnn_25/gru_cell_25/split/split_dim:output:0,backward_rnn_25/gru_cell_25/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2#
!backward_rnn_25/gru_cell_25/splité
3backward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOpReadVariableOp<backward_rnn_25_gru_cell_25_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype025
3backward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOpæ
$backward_rnn_25/gru_cell_25/MatMul_1MatMulbackward_rnn_25/zeros:output:0;backward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$backward_rnn_25/gru_cell_25/MatMul_1ê
%backward_rnn_25/gru_cell_25/BiasAdd_1BiasAdd.backward_rnn_25/gru_cell_25/MatMul_1:product:0,backward_rnn_25/gru_cell_25/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%backward_rnn_25/gru_cell_25/BiasAdd_1
!backward_rnn_25/gru_cell_25/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2#
!backward_rnn_25/gru_cell_25/Const©
-backward_rnn_25/gru_cell_25/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2/
-backward_rnn_25/gru_cell_25/split_1/split_dimà
#backward_rnn_25/gru_cell_25/split_1SplitV.backward_rnn_25/gru_cell_25/BiasAdd_1:output:0*backward_rnn_25/gru_cell_25/Const:output:06backward_rnn_25/gru_cell_25/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2%
#backward_rnn_25/gru_cell_25/split_1Ø
backward_rnn_25/gru_cell_25/addAddV2*backward_rnn_25/gru_cell_25/split:output:0,backward_rnn_25/gru_cell_25/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
backward_rnn_25/gru_cell_25/add­
#backward_rnn_25/gru_cell_25/SigmoidSigmoid#backward_rnn_25/gru_cell_25/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#backward_rnn_25/gru_cell_25/SigmoidÜ
!backward_rnn_25/gru_cell_25/add_1AddV2*backward_rnn_25/gru_cell_25/split:output:1,backward_rnn_25/gru_cell_25/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!backward_rnn_25/gru_cell_25/add_1³
%backward_rnn_25/gru_cell_25/Sigmoid_1Sigmoid%backward_rnn_25/gru_cell_25/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%backward_rnn_25/gru_cell_25/Sigmoid_1Õ
backward_rnn_25/gru_cell_25/mulMul)backward_rnn_25/gru_cell_25/Sigmoid_1:y:0,backward_rnn_25/gru_cell_25/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
backward_rnn_25/gru_cell_25/mulÓ
!backward_rnn_25/gru_cell_25/add_2AddV2*backward_rnn_25/gru_cell_25/split:output:2#backward_rnn_25/gru_cell_25/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!backward_rnn_25/gru_cell_25/add_2¦
 backward_rnn_25/gru_cell_25/TanhTanh%backward_rnn_25/gru_cell_25/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 backward_rnn_25/gru_cell_25/TanhÉ
!backward_rnn_25/gru_cell_25/mul_1Mul'backward_rnn_25/gru_cell_25/Sigmoid:y:0backward_rnn_25/zeros:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!backward_rnn_25/gru_cell_25/mul_1
!backward_rnn_25/gru_cell_25/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!backward_rnn_25/gru_cell_25/sub/xÑ
backward_rnn_25/gru_cell_25/subSub*backward_rnn_25/gru_cell_25/sub/x:output:0'backward_rnn_25/gru_cell_25/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
backward_rnn_25/gru_cell_25/subË
!backward_rnn_25/gru_cell_25/mul_2Mul#backward_rnn_25/gru_cell_25/sub:z:0$backward_rnn_25/gru_cell_25/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!backward_rnn_25/gru_cell_25/mul_2Ð
!backward_rnn_25/gru_cell_25/add_3AddV2%backward_rnn_25/gru_cell_25/mul_1:z:0%backward_rnn_25/gru_cell_25/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!backward_rnn_25/gru_cell_25/add_3¯
-backward_rnn_25/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2/
-backward_rnn_25/TensorArrayV2_1/element_shapeø
backward_rnn_25/TensorArrayV2_1TensorListReserve6backward_rnn_25/TensorArrayV2_1/element_shape:output:0(backward_rnn_25/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
backward_rnn_25/TensorArrayV2_1n
backward_rnn_25/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_rnn_25/time
(backward_rnn_25/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2*
(backward_rnn_25/while/maximum_iterations
"backward_rnn_25/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2$
"backward_rnn_25/while/loop_counter
backward_rnn_25/whileWhile+backward_rnn_25/while/loop_counter:output:01backward_rnn_25/while/maximum_iterations:output:0backward_rnn_25/time:output:0(backward_rnn_25/TensorArrayV2_1:handle:0backward_rnn_25/zeros:output:0(backward_rnn_25/strided_slice_1:output:0Gbackward_rnn_25/TensorArrayUnstack/TensorListFromTensor:output_handle:03backward_rnn_25_gru_cell_25_readvariableop_resource:backward_rnn_25_gru_cell_25_matmul_readvariableop_resource<backward_rnn_25_gru_cell_25_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*0
body(R&
$backward_rnn_25_while_body_387381917*0
cond(R&
$backward_rnn_25_while_cond_387381916*9
output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
backward_rnn_25/whileÕ
@backward_rnn_25/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2B
@backward_rnn_25/TensorArrayV2Stack/TensorListStack/element_shape²
2backward_rnn_25/TensorArrayV2Stack/TensorListStackTensorListStackbackward_rnn_25/while:output:3Ibackward_rnn_25/TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype024
2backward_rnn_25/TensorArrayV2Stack/TensorListStack¡
%backward_rnn_25/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2'
%backward_rnn_25/strided_slice_3/stack
'backward_rnn_25/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_rnn_25/strided_slice_3/stack_1
'backward_rnn_25/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_rnn_25/strided_slice_3/stack_2û
backward_rnn_25/strided_slice_3StridedSlice;backward_rnn_25/TensorArrayV2Stack/TensorListStack:tensor:0.backward_rnn_25/strided_slice_3/stack:output:00backward_rnn_25/strided_slice_3/stack_1:output:00backward_rnn_25/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2!
backward_rnn_25/strided_slice_3
 backward_rnn_25/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 backward_rnn_25/transpose_1/permï
backward_rnn_25/transpose_1	Transpose;backward_rnn_25/TensorArrayV2Stack/TensorListStack:tensor:0)backward_rnn_25/transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
backward_rnn_25/transpose_1\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisÁ
concatConcatV2'forward_rnn_25/strided_slice_3:output:0(backward_rnn_25/strided_slice_3:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
concat¾
IdentityIdentityconcat:output:02^backward_rnn_25/gru_cell_25/MatMul/ReadVariableOp4^backward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp+^backward_rnn_25/gru_cell_25/ReadVariableOp^backward_rnn_25/while1^forward_rnn_25/gru_cell_25/MatMul/ReadVariableOp3^forward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp*^forward_rnn_25/gru_cell_25/ReadVariableOp^forward_rnn_25/while*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2f
1backward_rnn_25/gru_cell_25/MatMul/ReadVariableOp1backward_rnn_25/gru_cell_25/MatMul/ReadVariableOp2j
3backward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp3backward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp2X
*backward_rnn_25/gru_cell_25/ReadVariableOp*backward_rnn_25/gru_cell_25/ReadVariableOp2.
backward_rnn_25/whilebackward_rnn_25/while2d
0forward_rnn_25/gru_cell_25/MatMul/ReadVariableOp0forward_rnn_25/gru_cell_25/MatMul/ReadVariableOp2h
2forward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp2forward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp2V
)forward_rnn_25/gru_cell_25/ReadVariableOp)forward_rnn_25/gru_cell_25/ReadVariableOp2,
forward_rnn_25/whileforward_rnn_25/while:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
H

while_body_387382611
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_25_readvariableop_resource_0:	F
2while_gru_cell_25_matmul_readvariableop_resource_0:
H
4while_gru_cell_25_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_25_readvariableop_resource:	D
0while_gru_cell_25_matmul_readvariableop_resource:
F
2while_gru_cell_25_matmul_1_readvariableop_resource:
¢'while/gru_cell_25/MatMul/ReadVariableOp¢)while/gru_cell_25/MatMul_1/ReadVariableOp¢ while/gru_cell_25/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem±
 while/gru_cell_25/ReadVariableOpReadVariableOp+while_gru_cell_25_readvariableop_resource_0*
_output_shapes
:	*
dtype02"
 while/gru_cell_25/ReadVariableOp¢
while/gru_cell_25/unstackUnpack(while/gru_cell_25/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
while/gru_cell_25/unstackÇ
'while/gru_cell_25/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_25_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype02)
'while/gru_cell_25/MatMul/ReadVariableOpÔ
while/gru_cell_25/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/MatMul¼
while/gru_cell_25/BiasAddBiasAdd"while/gru_cell_25/MatMul:product:0"while/gru_cell_25/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/BiasAdd
!while/gru_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2#
!while/gru_cell_25/split/split_dim÷
while/gru_cell_25/splitSplit*while/gru_cell_25/split/split_dim:output:0"while/gru_cell_25/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/gru_cell_25/splitÍ
)while/gru_cell_25/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_25_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02+
)while/gru_cell_25/MatMul_1/ReadVariableOp½
while/gru_cell_25/MatMul_1MatMulwhile_placeholder_21while/gru_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/MatMul_1Â
while/gru_cell_25/BiasAdd_1BiasAdd$while/gru_cell_25/MatMul_1:product:0"while/gru_cell_25/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/BiasAdd_1
while/gru_cell_25/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2
while/gru_cell_25/Const
#while/gru_cell_25/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#while/gru_cell_25/split_1/split_dim®
while/gru_cell_25/split_1SplitV$while/gru_cell_25/BiasAdd_1:output:0 while/gru_cell_25/Const:output:0,while/gru_cell_25/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/gru_cell_25/split_1°
while/gru_cell_25/addAddV2 while/gru_cell_25/split:output:0"while/gru_cell_25/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/add
while/gru_cell_25/SigmoidSigmoidwhile/gru_cell_25/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/Sigmoid´
while/gru_cell_25/add_1AddV2 while/gru_cell_25/split:output:1"while/gru_cell_25/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/add_1
while/gru_cell_25/Sigmoid_1Sigmoidwhile/gru_cell_25/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/Sigmoid_1­
while/gru_cell_25/mulMulwhile/gru_cell_25/Sigmoid_1:y:0"while/gru_cell_25/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/mul«
while/gru_cell_25/add_2AddV2 while/gru_cell_25/split:output:2while/gru_cell_25/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/add_2
while/gru_cell_25/TanhTanhwhile/gru_cell_25/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/Tanh 
while/gru_cell_25/mul_1Mulwhile/gru_cell_25/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/mul_1w
while/gru_cell_25/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_25/sub/x©
while/gru_cell_25/subSub while/gru_cell_25/sub/x:output:0while/gru_cell_25/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/sub£
while/gru_cell_25/mul_2Mulwhile/gru_cell_25/sub:z:0while/gru_cell_25/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/mul_2¨
while/gru_cell_25/add_3AddV2while/gru_cell_25/mul_1:z:0while/gru_cell_25/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/add_3ß
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_25/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1×
while/IdentityIdentitywhile/add_1:z:0(^while/gru_cell_25/MatMul/ReadVariableOp*^while/gru_cell_25/MatMul_1/ReadVariableOp!^while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityê
while/Identity_1Identitywhile_while_maximum_iterations(^while/gru_cell_25/MatMul/ReadVariableOp*^while/gru_cell_25/MatMul_1/ReadVariableOp!^while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1Ù
while/Identity_2Identitywhile/add:z:0(^while/gru_cell_25/MatMul/ReadVariableOp*^while/gru_cell_25/MatMul_1/ReadVariableOp!^while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/gru_cell_25/MatMul/ReadVariableOp*^while/gru_cell_25/MatMul_1/ReadVariableOp!^while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3ù
while/Identity_4Identitywhile/gru_cell_25/add_3:z:0(^while/gru_cell_25/MatMul/ReadVariableOp*^while/gru_cell_25/MatMul_1/ReadVariableOp!^while/gru_cell_25/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4"j
2while_gru_cell_25_matmul_1_readvariableop_resource4while_gru_cell_25_matmul_1_readvariableop_resource_0"f
0while_gru_cell_25_matmul_readvariableop_resource2while_gru_cell_25_matmul_readvariableop_resource_0"X
)while_gru_cell_25_readvariableop_resource+while_gru_cell_25_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : 2R
'while/gru_cell_25/MatMul/ReadVariableOp'while/gru_cell_25/MatMul/ReadVariableOp2V
)while/gru_cell_25/MatMul_1/ReadVariableOp)while/gru_cell_25/MatMul_1/ReadVariableOp2D
 while/gru_cell_25/ReadVariableOp while/gru_cell_25/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 

¹	
<Double_Bidirectional_GRU_forward_rnn_25_while_cond_387380743l
hdouble_bidirectional_gru_forward_rnn_25_while_double_bidirectional_gru_forward_rnn_25_while_loop_counterr
ndouble_bidirectional_gru_forward_rnn_25_while_double_bidirectional_gru_forward_rnn_25_while_maximum_iterations=
9double_bidirectional_gru_forward_rnn_25_while_placeholder?
;double_bidirectional_gru_forward_rnn_25_while_placeholder_1?
;double_bidirectional_gru_forward_rnn_25_while_placeholder_2n
jdouble_bidirectional_gru_forward_rnn_25_while_less_double_bidirectional_gru_forward_rnn_25_strided_slice_1
double_bidirectional_gru_forward_rnn_25_while_double_bidirectional_gru_forward_rnn_25_while_cond_387380743___redundant_placeholder0
double_bidirectional_gru_forward_rnn_25_while_double_bidirectional_gru_forward_rnn_25_while_cond_387380743___redundant_placeholder1
double_bidirectional_gru_forward_rnn_25_while_double_bidirectional_gru_forward_rnn_25_while_cond_387380743___redundant_placeholder2
double_bidirectional_gru_forward_rnn_25_while_double_bidirectional_gru_forward_rnn_25_while_cond_387380743___redundant_placeholder3:
6double_bidirectional_gru_forward_rnn_25_while_identity
¸
2Double_Bidirectional_GRU/forward_rnn_25/while/LessLess9double_bidirectional_gru_forward_rnn_25_while_placeholderjdouble_bidirectional_gru_forward_rnn_25_while_less_double_bidirectional_gru_forward_rnn_25_strided_slice_1*
T0*
_output_shapes
: 24
2Double_Bidirectional_GRU/forward_rnn_25/while/LessÕ
6Double_Bidirectional_GRU/forward_rnn_25/while/IdentityIdentity6Double_Bidirectional_GRU/forward_rnn_25/while/Less:z:0*
T0
*
_output_shapes
: 28
6Double_Bidirectional_GRU/forward_rnn_25/while/Identity"y
6double_bidirectional_gru_forward_rnn_25_while_identity?Double_Bidirectional_GRU/forward_rnn_25/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ª#

"__inference__traced_save_387382958
file_prefix0
,savev2_prediction_kernel_read_readvariableop.
*savev2_prediction_bias_read_readvariableopY
Usavev2_double_bidirectional_gru_forward_rnn_25_gru_cell_25_kernel_read_readvariableopc
_savev2_double_bidirectional_gru_forward_rnn_25_gru_cell_25_recurrent_kernel_read_readvariableopW
Ssavev2_double_bidirectional_gru_forward_rnn_25_gru_cell_25_bias_read_readvariableopZ
Vsavev2_double_bidirectional_gru_backward_rnn_25_gru_cell_25_kernel_read_readvariableopd
`savev2_double_bidirectional_gru_backward_rnn_25_gru_cell_25_recurrent_kernel_read_readvariableopX
Tsavev2_double_bidirectional_gru_backward_rnn_25_gru_cell_25_bias_read_readvariableop
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
ShardedFilename»
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*Í
valueÃBÀ	B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
SaveV2/shape_and_slices¹
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_prediction_kernel_read_readvariableop*savev2_prediction_bias_read_readvariableopUsavev2_double_bidirectional_gru_forward_rnn_25_gru_cell_25_kernel_read_readvariableop_savev2_double_bidirectional_gru_forward_rnn_25_gru_cell_25_recurrent_kernel_read_readvariableopSsavev2_double_bidirectional_gru_forward_rnn_25_gru_cell_25_bias_read_readvariableopVsavev2_double_bidirectional_gru_backward_rnn_25_gru_cell_25_kernel_read_readvariableop`savev2_double_bidirectional_gru_backward_rnn_25_gru_cell_25_recurrent_kernel_read_readvariableopTsavev2_double_bidirectional_gru_backward_rnn_25_gru_cell_25_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
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

identity_1Identity_1:output:0*n
_input_shapes]
[: :	::
:
:	:
:
:	: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	: 

_output_shapes
::&"
 
_output_shapes
:
:&"
 
_output_shapes
:
:%!

_output_shapes
:	:&"
 
_output_shapes
:
:&"
 
_output_shapes
:
:%!

_output_shapes
:	:	

_output_shapes
: 
µ
n
P__inference_Embedding_Dropout_layer_call_and_return_conditional_losses_387381337

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
ñ
¡
W__inference_Double_Bidirectional_GRU_layer_call_and_return_conditional_losses_387380440

inputsE
2forward_rnn_25_gru_cell_25_readvariableop_resource:	M
9forward_rnn_25_gru_cell_25_matmul_readvariableop_resource:
O
;forward_rnn_25_gru_cell_25_matmul_1_readvariableop_resource:
F
3backward_rnn_25_gru_cell_25_readvariableop_resource:	N
:backward_rnn_25_gru_cell_25_matmul_readvariableop_resource:
P
<backward_rnn_25_gru_cell_25_matmul_1_readvariableop_resource:

identity¢1backward_rnn_25/gru_cell_25/MatMul/ReadVariableOp¢3backward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp¢*backward_rnn_25/gru_cell_25/ReadVariableOp¢backward_rnn_25/while¢0forward_rnn_25/gru_cell_25/MatMul/ReadVariableOp¢2forward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp¢)forward_rnn_25/gru_cell_25/ReadVariableOp¢forward_rnn_25/whileb
forward_rnn_25/ShapeShapeinputs*
T0*
_output_shapes
:2
forward_rnn_25/Shape
"forward_rnn_25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"forward_rnn_25/strided_slice/stack
$forward_rnn_25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$forward_rnn_25/strided_slice/stack_1
$forward_rnn_25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$forward_rnn_25/strided_slice/stack_2¼
forward_rnn_25/strided_sliceStridedSliceforward_rnn_25/Shape:output:0+forward_rnn_25/strided_slice/stack:output:0-forward_rnn_25/strided_slice/stack_1:output:0-forward_rnn_25/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
forward_rnn_25/strided_slice{
forward_rnn_25/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
forward_rnn_25/zeros/mul/y¨
forward_rnn_25/zeros/mulMul%forward_rnn_25/strided_slice:output:0#forward_rnn_25/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_rnn_25/zeros/mul}
forward_rnn_25/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
forward_rnn_25/zeros/Less/y£
forward_rnn_25/zeros/LessLessforward_rnn_25/zeros/mul:z:0$forward_rnn_25/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_rnn_25/zeros/Less
forward_rnn_25/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
forward_rnn_25/zeros/packed/1¿
forward_rnn_25/zeros/packedPack%forward_rnn_25/strided_slice:output:0&forward_rnn_25/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_rnn_25/zeros/packed}
forward_rnn_25/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_rnn_25/zeros/Const²
forward_rnn_25/zerosFill$forward_rnn_25/zeros/packed:output:0#forward_rnn_25/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
forward_rnn_25/zeros
forward_rnn_25/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
forward_rnn_25/transpose/perm±
forward_rnn_25/transpose	Transposeinputs&forward_rnn_25/transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
forward_rnn_25/transpose|
forward_rnn_25/Shape_1Shapeforward_rnn_25/transpose:y:0*
T0*
_output_shapes
:2
forward_rnn_25/Shape_1
$forward_rnn_25/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_rnn_25/strided_slice_1/stack
&forward_rnn_25/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_rnn_25/strided_slice_1/stack_1
&forward_rnn_25/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_rnn_25/strided_slice_1/stack_2È
forward_rnn_25/strided_slice_1StridedSliceforward_rnn_25/Shape_1:output:0-forward_rnn_25/strided_slice_1/stack:output:0/forward_rnn_25/strided_slice_1/stack_1:output:0/forward_rnn_25/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
forward_rnn_25/strided_slice_1£
*forward_rnn_25/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2,
*forward_rnn_25/TensorArrayV2/element_shapeî
forward_rnn_25/TensorArrayV2TensorListReserve3forward_rnn_25/TensorArrayV2/element_shape:output:0'forward_rnn_25/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
forward_rnn_25/TensorArrayV2Ý
Dforward_rnn_25/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2F
Dforward_rnn_25/TensorArrayUnstack/TensorListFromTensor/element_shape´
6forward_rnn_25/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_rnn_25/transpose:y:0Mforward_rnn_25/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type028
6forward_rnn_25/TensorArrayUnstack/TensorListFromTensor
$forward_rnn_25/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_rnn_25/strided_slice_2/stack
&forward_rnn_25/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_rnn_25/strided_slice_2/stack_1
&forward_rnn_25/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_rnn_25/strided_slice_2/stack_2×
forward_rnn_25/strided_slice_2StridedSliceforward_rnn_25/transpose:y:0-forward_rnn_25/strided_slice_2/stack:output:0/forward_rnn_25/strided_slice_2/stack_1:output:0/forward_rnn_25/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2 
forward_rnn_25/strided_slice_2Ê
)forward_rnn_25/gru_cell_25/ReadVariableOpReadVariableOp2forward_rnn_25_gru_cell_25_readvariableop_resource*
_output_shapes
:	*
dtype02+
)forward_rnn_25/gru_cell_25/ReadVariableOp½
"forward_rnn_25/gru_cell_25/unstackUnpack1forward_rnn_25/gru_cell_25/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2$
"forward_rnn_25/gru_cell_25/unstackà
0forward_rnn_25/gru_cell_25/MatMul/ReadVariableOpReadVariableOp9forward_rnn_25_gru_cell_25_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype022
0forward_rnn_25/gru_cell_25/MatMul/ReadVariableOpæ
!forward_rnn_25/gru_cell_25/MatMulMatMul'forward_rnn_25/strided_slice_2:output:08forward_rnn_25/gru_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!forward_rnn_25/gru_cell_25/MatMulà
"forward_rnn_25/gru_cell_25/BiasAddBiasAdd+forward_rnn_25/gru_cell_25/MatMul:product:0+forward_rnn_25/gru_cell_25/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"forward_rnn_25/gru_cell_25/BiasAdd£
*forward_rnn_25/gru_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2,
*forward_rnn_25/gru_cell_25/split/split_dim
 forward_rnn_25/gru_cell_25/splitSplit3forward_rnn_25/gru_cell_25/split/split_dim:output:0+forward_rnn_25/gru_cell_25/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2"
 forward_rnn_25/gru_cell_25/splitæ
2forward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOpReadVariableOp;forward_rnn_25_gru_cell_25_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype024
2forward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOpâ
#forward_rnn_25/gru_cell_25/MatMul_1MatMulforward_rnn_25/zeros:output:0:forward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#forward_rnn_25/gru_cell_25/MatMul_1æ
$forward_rnn_25/gru_cell_25/BiasAdd_1BiasAdd-forward_rnn_25/gru_cell_25/MatMul_1:product:0+forward_rnn_25/gru_cell_25/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$forward_rnn_25/gru_cell_25/BiasAdd_1
 forward_rnn_25/gru_cell_25/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2"
 forward_rnn_25/gru_cell_25/Const§
,forward_rnn_25/gru_cell_25/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2.
,forward_rnn_25/gru_cell_25/split_1/split_dimÛ
"forward_rnn_25/gru_cell_25/split_1SplitV-forward_rnn_25/gru_cell_25/BiasAdd_1:output:0)forward_rnn_25/gru_cell_25/Const:output:05forward_rnn_25/gru_cell_25/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2$
"forward_rnn_25/gru_cell_25/split_1Ô
forward_rnn_25/gru_cell_25/addAddV2)forward_rnn_25/gru_cell_25/split:output:0+forward_rnn_25/gru_cell_25/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
forward_rnn_25/gru_cell_25/addª
"forward_rnn_25/gru_cell_25/SigmoidSigmoid"forward_rnn_25/gru_cell_25/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"forward_rnn_25/gru_cell_25/SigmoidØ
 forward_rnn_25/gru_cell_25/add_1AddV2)forward_rnn_25/gru_cell_25/split:output:1+forward_rnn_25/gru_cell_25/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 forward_rnn_25/gru_cell_25/add_1°
$forward_rnn_25/gru_cell_25/Sigmoid_1Sigmoid$forward_rnn_25/gru_cell_25/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$forward_rnn_25/gru_cell_25/Sigmoid_1Ñ
forward_rnn_25/gru_cell_25/mulMul(forward_rnn_25/gru_cell_25/Sigmoid_1:y:0+forward_rnn_25/gru_cell_25/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
forward_rnn_25/gru_cell_25/mulÏ
 forward_rnn_25/gru_cell_25/add_2AddV2)forward_rnn_25/gru_cell_25/split:output:2"forward_rnn_25/gru_cell_25/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 forward_rnn_25/gru_cell_25/add_2£
forward_rnn_25/gru_cell_25/TanhTanh$forward_rnn_25/gru_cell_25/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
forward_rnn_25/gru_cell_25/TanhÅ
 forward_rnn_25/gru_cell_25/mul_1Mul&forward_rnn_25/gru_cell_25/Sigmoid:y:0forward_rnn_25/zeros:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 forward_rnn_25/gru_cell_25/mul_1
 forward_rnn_25/gru_cell_25/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2"
 forward_rnn_25/gru_cell_25/sub/xÍ
forward_rnn_25/gru_cell_25/subSub)forward_rnn_25/gru_cell_25/sub/x:output:0&forward_rnn_25/gru_cell_25/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
forward_rnn_25/gru_cell_25/subÇ
 forward_rnn_25/gru_cell_25/mul_2Mul"forward_rnn_25/gru_cell_25/sub:z:0#forward_rnn_25/gru_cell_25/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 forward_rnn_25/gru_cell_25/mul_2Ì
 forward_rnn_25/gru_cell_25/add_3AddV2$forward_rnn_25/gru_cell_25/mul_1:z:0$forward_rnn_25/gru_cell_25/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 forward_rnn_25/gru_cell_25/add_3­
,forward_rnn_25/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2.
,forward_rnn_25/TensorArrayV2_1/element_shapeô
forward_rnn_25/TensorArrayV2_1TensorListReserve5forward_rnn_25/TensorArrayV2_1/element_shape:output:0'forward_rnn_25/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
forward_rnn_25/TensorArrayV2_1l
forward_rnn_25/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_rnn_25/time
'forward_rnn_25/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2)
'forward_rnn_25/while/maximum_iterations
!forward_rnn_25/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2#
!forward_rnn_25/while/loop_counter
forward_rnn_25/whileWhile*forward_rnn_25/while/loop_counter:output:00forward_rnn_25/while/maximum_iterations:output:0forward_rnn_25/time:output:0'forward_rnn_25/TensorArrayV2_1:handle:0forward_rnn_25/zeros:output:0'forward_rnn_25/strided_slice_1:output:0Fforward_rnn_25/TensorArrayUnstack/TensorListFromTensor:output_handle:02forward_rnn_25_gru_cell_25_readvariableop_resource9forward_rnn_25_gru_cell_25_matmul_readvariableop_resource;forward_rnn_25_gru_cell_25_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*/
body'R%
#forward_rnn_25_while_body_387380196*/
cond'R%
#forward_rnn_25_while_cond_387380195*9
output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
forward_rnn_25/whileÓ
?forward_rnn_25/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2A
?forward_rnn_25/TensorArrayV2Stack/TensorListStack/element_shape®
1forward_rnn_25/TensorArrayV2Stack/TensorListStackTensorListStackforward_rnn_25/while:output:3Hforward_rnn_25/TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype023
1forward_rnn_25/TensorArrayV2Stack/TensorListStack
$forward_rnn_25/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2&
$forward_rnn_25/strided_slice_3/stack
&forward_rnn_25/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_rnn_25/strided_slice_3/stack_1
&forward_rnn_25/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_rnn_25/strided_slice_3/stack_2õ
forward_rnn_25/strided_slice_3StridedSlice:forward_rnn_25/TensorArrayV2Stack/TensorListStack:tensor:0-forward_rnn_25/strided_slice_3/stack:output:0/forward_rnn_25/strided_slice_3/stack_1:output:0/forward_rnn_25/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2 
forward_rnn_25/strided_slice_3
forward_rnn_25/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
forward_rnn_25/transpose_1/permë
forward_rnn_25/transpose_1	Transpose:forward_rnn_25/TensorArrayV2Stack/TensorListStack:tensor:0(forward_rnn_25/transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
forward_rnn_25/transpose_1d
backward_rnn_25/ShapeShapeinputs*
T0*
_output_shapes
:2
backward_rnn_25/Shape
#backward_rnn_25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#backward_rnn_25/strided_slice/stack
%backward_rnn_25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%backward_rnn_25/strided_slice/stack_1
%backward_rnn_25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%backward_rnn_25/strided_slice/stack_2Â
backward_rnn_25/strided_sliceStridedSlicebackward_rnn_25/Shape:output:0,backward_rnn_25/strided_slice/stack:output:0.backward_rnn_25/strided_slice/stack_1:output:0.backward_rnn_25/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
backward_rnn_25/strided_slice}
backward_rnn_25/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
backward_rnn_25/zeros/mul/y¬
backward_rnn_25/zeros/mulMul&backward_rnn_25/strided_slice:output:0$backward_rnn_25/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_rnn_25/zeros/mul
backward_rnn_25/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
backward_rnn_25/zeros/Less/y§
backward_rnn_25/zeros/LessLessbackward_rnn_25/zeros/mul:z:0%backward_rnn_25/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_rnn_25/zeros/Less
backward_rnn_25/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2 
backward_rnn_25/zeros/packed/1Ã
backward_rnn_25/zeros/packedPack&backward_rnn_25/strided_slice:output:0'backward_rnn_25/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
backward_rnn_25/zeros/packed
backward_rnn_25/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_rnn_25/zeros/Const¶
backward_rnn_25/zerosFill%backward_rnn_25/zeros/packed:output:0$backward_rnn_25/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
backward_rnn_25/zeros
backward_rnn_25/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
backward_rnn_25/transpose/perm´
backward_rnn_25/transpose	Transposeinputs'backward_rnn_25/transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
backward_rnn_25/transpose
backward_rnn_25/Shape_1Shapebackward_rnn_25/transpose:y:0*
T0*
_output_shapes
:2
backward_rnn_25/Shape_1
%backward_rnn_25/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_rnn_25/strided_slice_1/stack
'backward_rnn_25/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_rnn_25/strided_slice_1/stack_1
'backward_rnn_25/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_rnn_25/strided_slice_1/stack_2Î
backward_rnn_25/strided_slice_1StridedSlice backward_rnn_25/Shape_1:output:0.backward_rnn_25/strided_slice_1/stack:output:00backward_rnn_25/strided_slice_1/stack_1:output:00backward_rnn_25/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
backward_rnn_25/strided_slice_1¥
+backward_rnn_25/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2-
+backward_rnn_25/TensorArrayV2/element_shapeò
backward_rnn_25/TensorArrayV2TensorListReserve4backward_rnn_25/TensorArrayV2/element_shape:output:0(backward_rnn_25/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
backward_rnn_25/TensorArrayV2
backward_rnn_25/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2 
backward_rnn_25/ReverseV2/axisË
backward_rnn_25/ReverseV2	ReverseV2backward_rnn_25/transpose:y:0'backward_rnn_25/ReverseV2/axis:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
backward_rnn_25/ReverseV2ß
Ebackward_rnn_25/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2G
Ebackward_rnn_25/TensorArrayUnstack/TensorListFromTensor/element_shape½
7backward_rnn_25/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"backward_rnn_25/ReverseV2:output:0Nbackward_rnn_25/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7backward_rnn_25/TensorArrayUnstack/TensorListFromTensor
%backward_rnn_25/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_rnn_25/strided_slice_2/stack
'backward_rnn_25/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_rnn_25/strided_slice_2/stack_1
'backward_rnn_25/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_rnn_25/strided_slice_2/stack_2Ý
backward_rnn_25/strided_slice_2StridedSlicebackward_rnn_25/transpose:y:0.backward_rnn_25/strided_slice_2/stack:output:00backward_rnn_25/strided_slice_2/stack_1:output:00backward_rnn_25/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2!
backward_rnn_25/strided_slice_2Í
*backward_rnn_25/gru_cell_25/ReadVariableOpReadVariableOp3backward_rnn_25_gru_cell_25_readvariableop_resource*
_output_shapes
:	*
dtype02,
*backward_rnn_25/gru_cell_25/ReadVariableOpÀ
#backward_rnn_25/gru_cell_25/unstackUnpack2backward_rnn_25/gru_cell_25/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2%
#backward_rnn_25/gru_cell_25/unstackã
1backward_rnn_25/gru_cell_25/MatMul/ReadVariableOpReadVariableOp:backward_rnn_25_gru_cell_25_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype023
1backward_rnn_25/gru_cell_25/MatMul/ReadVariableOpê
"backward_rnn_25/gru_cell_25/MatMulMatMul(backward_rnn_25/strided_slice_2:output:09backward_rnn_25/gru_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"backward_rnn_25/gru_cell_25/MatMulä
#backward_rnn_25/gru_cell_25/BiasAddBiasAdd,backward_rnn_25/gru_cell_25/MatMul:product:0,backward_rnn_25/gru_cell_25/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#backward_rnn_25/gru_cell_25/BiasAdd¥
+backward_rnn_25/gru_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2-
+backward_rnn_25/gru_cell_25/split/split_dim
!backward_rnn_25/gru_cell_25/splitSplit4backward_rnn_25/gru_cell_25/split/split_dim:output:0,backward_rnn_25/gru_cell_25/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2#
!backward_rnn_25/gru_cell_25/splité
3backward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOpReadVariableOp<backward_rnn_25_gru_cell_25_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype025
3backward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOpæ
$backward_rnn_25/gru_cell_25/MatMul_1MatMulbackward_rnn_25/zeros:output:0;backward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$backward_rnn_25/gru_cell_25/MatMul_1ê
%backward_rnn_25/gru_cell_25/BiasAdd_1BiasAdd.backward_rnn_25/gru_cell_25/MatMul_1:product:0,backward_rnn_25/gru_cell_25/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%backward_rnn_25/gru_cell_25/BiasAdd_1
!backward_rnn_25/gru_cell_25/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2#
!backward_rnn_25/gru_cell_25/Const©
-backward_rnn_25/gru_cell_25/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2/
-backward_rnn_25/gru_cell_25/split_1/split_dimà
#backward_rnn_25/gru_cell_25/split_1SplitV.backward_rnn_25/gru_cell_25/BiasAdd_1:output:0*backward_rnn_25/gru_cell_25/Const:output:06backward_rnn_25/gru_cell_25/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2%
#backward_rnn_25/gru_cell_25/split_1Ø
backward_rnn_25/gru_cell_25/addAddV2*backward_rnn_25/gru_cell_25/split:output:0,backward_rnn_25/gru_cell_25/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
backward_rnn_25/gru_cell_25/add­
#backward_rnn_25/gru_cell_25/SigmoidSigmoid#backward_rnn_25/gru_cell_25/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#backward_rnn_25/gru_cell_25/SigmoidÜ
!backward_rnn_25/gru_cell_25/add_1AddV2*backward_rnn_25/gru_cell_25/split:output:1,backward_rnn_25/gru_cell_25/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!backward_rnn_25/gru_cell_25/add_1³
%backward_rnn_25/gru_cell_25/Sigmoid_1Sigmoid%backward_rnn_25/gru_cell_25/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%backward_rnn_25/gru_cell_25/Sigmoid_1Õ
backward_rnn_25/gru_cell_25/mulMul)backward_rnn_25/gru_cell_25/Sigmoid_1:y:0,backward_rnn_25/gru_cell_25/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
backward_rnn_25/gru_cell_25/mulÓ
!backward_rnn_25/gru_cell_25/add_2AddV2*backward_rnn_25/gru_cell_25/split:output:2#backward_rnn_25/gru_cell_25/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!backward_rnn_25/gru_cell_25/add_2¦
 backward_rnn_25/gru_cell_25/TanhTanh%backward_rnn_25/gru_cell_25/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 backward_rnn_25/gru_cell_25/TanhÉ
!backward_rnn_25/gru_cell_25/mul_1Mul'backward_rnn_25/gru_cell_25/Sigmoid:y:0backward_rnn_25/zeros:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!backward_rnn_25/gru_cell_25/mul_1
!backward_rnn_25/gru_cell_25/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!backward_rnn_25/gru_cell_25/sub/xÑ
backward_rnn_25/gru_cell_25/subSub*backward_rnn_25/gru_cell_25/sub/x:output:0'backward_rnn_25/gru_cell_25/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
backward_rnn_25/gru_cell_25/subË
!backward_rnn_25/gru_cell_25/mul_2Mul#backward_rnn_25/gru_cell_25/sub:z:0$backward_rnn_25/gru_cell_25/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!backward_rnn_25/gru_cell_25/mul_2Ð
!backward_rnn_25/gru_cell_25/add_3AddV2%backward_rnn_25/gru_cell_25/mul_1:z:0%backward_rnn_25/gru_cell_25/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!backward_rnn_25/gru_cell_25/add_3¯
-backward_rnn_25/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2/
-backward_rnn_25/TensorArrayV2_1/element_shapeø
backward_rnn_25/TensorArrayV2_1TensorListReserve6backward_rnn_25/TensorArrayV2_1/element_shape:output:0(backward_rnn_25/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
backward_rnn_25/TensorArrayV2_1n
backward_rnn_25/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_rnn_25/time
(backward_rnn_25/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2*
(backward_rnn_25/while/maximum_iterations
"backward_rnn_25/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2$
"backward_rnn_25/while/loop_counter
backward_rnn_25/whileWhile+backward_rnn_25/while/loop_counter:output:01backward_rnn_25/while/maximum_iterations:output:0backward_rnn_25/time:output:0(backward_rnn_25/TensorArrayV2_1:handle:0backward_rnn_25/zeros:output:0(backward_rnn_25/strided_slice_1:output:0Gbackward_rnn_25/TensorArrayUnstack/TensorListFromTensor:output_handle:03backward_rnn_25_gru_cell_25_readvariableop_resource:backward_rnn_25_gru_cell_25_matmul_readvariableop_resource<backward_rnn_25_gru_cell_25_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*0
body(R&
$backward_rnn_25_while_body_387380350*0
cond(R&
$backward_rnn_25_while_cond_387380349*9
output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
backward_rnn_25/whileÕ
@backward_rnn_25/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2B
@backward_rnn_25/TensorArrayV2Stack/TensorListStack/element_shape²
2backward_rnn_25/TensorArrayV2Stack/TensorListStackTensorListStackbackward_rnn_25/while:output:3Ibackward_rnn_25/TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype024
2backward_rnn_25/TensorArrayV2Stack/TensorListStack¡
%backward_rnn_25/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2'
%backward_rnn_25/strided_slice_3/stack
'backward_rnn_25/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_rnn_25/strided_slice_3/stack_1
'backward_rnn_25/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_rnn_25/strided_slice_3/stack_2û
backward_rnn_25/strided_slice_3StridedSlice;backward_rnn_25/TensorArrayV2Stack/TensorListStack:tensor:0.backward_rnn_25/strided_slice_3/stack:output:00backward_rnn_25/strided_slice_3/stack_1:output:00backward_rnn_25/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2!
backward_rnn_25/strided_slice_3
 backward_rnn_25/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 backward_rnn_25/transpose_1/permï
backward_rnn_25/transpose_1	Transpose;backward_rnn_25/TensorArrayV2Stack/TensorListStack:tensor:0)backward_rnn_25/transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
backward_rnn_25/transpose_1\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisÁ
concatConcatV2'forward_rnn_25/strided_slice_3:output:0(backward_rnn_25/strided_slice_3:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
concat¾
IdentityIdentityconcat:output:02^backward_rnn_25/gru_cell_25/MatMul/ReadVariableOp4^backward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp+^backward_rnn_25/gru_cell_25/ReadVariableOp^backward_rnn_25/while1^forward_rnn_25/gru_cell_25/MatMul/ReadVariableOp3^forward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp*^forward_rnn_25/gru_cell_25/ReadVariableOp^forward_rnn_25/while*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2f
1backward_rnn_25/gru_cell_25/MatMul/ReadVariableOp1backward_rnn_25/gru_cell_25/MatMul/ReadVariableOp2j
3backward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp3backward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp2X
*backward_rnn_25/gru_cell_25/ReadVariableOp*backward_rnn_25/gru_cell_25/ReadVariableOp2.
backward_rnn_25/whilebackward_rnn_25/while2d
0forward_rnn_25/gru_cell_25/MatMul/ReadVariableOp0forward_rnn_25/gru_cell_25/MatMul/ReadVariableOp2h
2forward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp2forward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp2V
)forward_rnn_25/gru_cell_25/ReadVariableOp)forward_rnn_25/gru_cell_25/ReadVariableOp2,
forward_rnn_25/whileforward_rnn_25/while:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
?

N__inference_backward_rnn_25_layer_call_and_return_conditional_losses_387379287

inputs(
gru_cell_25_387379212:	)
gru_cell_25_387379214:
)
gru_cell_25_387379216:

identity¢#gru_cell_25/StatefulPartitionedCall¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2j
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2
ReverseV2/axis
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	ReverseV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeý
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2
#gru_cell_25/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_25_387379212gru_cell_25_387379214gru_cell_25_387379216*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_gru_cell_25_layer_call_and_return_conditional_losses_3873792112%
#gru_cell_25/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterø
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_25_387379212gru_cell_25_387379214gru_cell_25_387379216*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	* 
bodyR
while_body_387379224* 
condR
while_cond_387379223*9
output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
transpose_1
IdentityIdentitystrided_slice_3:output:0$^gru_cell_25/StatefulPartitionedCall^while*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2J
#gru_cell_25/StatefulPartitionedCall#gru_cell_25/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Ç
2__inference_forward_rnn_25_layer_call_fn_387382038
inputs_0
unknown:	
	unknown_0:

	unknown_1:

identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_forward_rnn_25_layer_call_and_return_conditional_losses_3873787072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
À
Ö
#forward_rnn_25_while_cond_387379802:
6forward_rnn_25_while_forward_rnn_25_while_loop_counter@
<forward_rnn_25_while_forward_rnn_25_while_maximum_iterations$
 forward_rnn_25_while_placeholder&
"forward_rnn_25_while_placeholder_1&
"forward_rnn_25_while_placeholder_2<
8forward_rnn_25_while_less_forward_rnn_25_strided_slice_1U
Qforward_rnn_25_while_forward_rnn_25_while_cond_387379802___redundant_placeholder0U
Qforward_rnn_25_while_forward_rnn_25_while_cond_387379802___redundant_placeholder1U
Qforward_rnn_25_while_forward_rnn_25_while_cond_387379802___redundant_placeholder2U
Qforward_rnn_25_while_forward_rnn_25_while_cond_387379802___redundant_placeholder3!
forward_rnn_25_while_identity
»
forward_rnn_25/while/LessLess forward_rnn_25_while_placeholder8forward_rnn_25_while_less_forward_rnn_25_strided_slice_1*
T0*
_output_shapes
: 2
forward_rnn_25/while/Less
forward_rnn_25/while/IdentityIdentityforward_rnn_25/while/Less:z:0*
T0
*
_output_shapes
: 2
forward_rnn_25/while/Identity"G
forward_rnn_25_while_identity&forward_rnn_25/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
×
Þ
J__inference_gru_cell_25_layer_call_and_return_conditional_losses_387379211

inputs

states*
readvariableop_resource:	2
matmul_readvariableop_resource:
4
 matmul_1_readvariableop_resource:

identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpy
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
split/split_dim¯
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	BiasAdd_1c
ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2
Constq
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
split_1/split_dimÔ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanh]
mul_1MulSigmoid:y:0states*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sub[
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_3¡
IdentityIdentity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¥

Identity_1Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
ï*
½
%__inference__traced_restore_387382992
file_prefix5
"assignvariableop_prediction_kernel:	0
"assignvariableop_1_prediction_bias:a
Massignvariableop_2_double_bidirectional_gru_forward_rnn_25_gru_cell_25_kernel:
k
Wassignvariableop_3_double_bidirectional_gru_forward_rnn_25_gru_cell_25_recurrent_kernel:
^
Kassignvariableop_4_double_bidirectional_gru_forward_rnn_25_gru_cell_25_bias:	b
Nassignvariableop_5_double_bidirectional_gru_backward_rnn_25_gru_cell_25_kernel:
l
Xassignvariableop_6_double_bidirectional_gru_backward_rnn_25_gru_cell_25_recurrent_kernel:
_
Lassignvariableop_7_double_bidirectional_gru_backward_rnn_25_gru_cell_25_bias:	

identity_9¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7Á
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*Í
valueÃBÀ	B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names 
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
RestoreV2/shape_and_slicesØ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*8
_output_shapes&
$:::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity¡
AssignVariableOpAssignVariableOp"assignvariableop_prediction_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1§
AssignVariableOp_1AssignVariableOp"assignvariableop_1_prediction_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ò
AssignVariableOp_2AssignVariableOpMassignvariableop_2_double_bidirectional_gru_forward_rnn_25_gru_cell_25_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ü
AssignVariableOp_3AssignVariableOpWassignvariableop_3_double_bidirectional_gru_forward_rnn_25_gru_cell_25_recurrent_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Ð
AssignVariableOp_4AssignVariableOpKassignvariableop_4_double_bidirectional_gru_forward_rnn_25_gru_cell_25_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Ó
AssignVariableOp_5AssignVariableOpNassignvariableop_5_double_bidirectional_gru_backward_rnn_25_gru_cell_25_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6Ý
AssignVariableOp_6AssignVariableOpXassignvariableop_6_double_bidirectional_gru_backward_rnn_25_gru_cell_25_recurrent_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ñ
AssignVariableOp_7AssignVariableOpLassignvariableop_7_double_bidirectional_gru_backward_rnn_25_gru_cell_25_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp

Identity_8Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_8

Identity_9IdentityIdentity_8:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7*
T0*
_output_shapes
: 2

Identity_9"!

identity_9Identity_9:output:0*%
_input_shapes
: : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_7:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
·

û
I__inference_Prediction_layer_call_and_return_conditional_losses_387382027

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
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
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»
É
JTree_Encoder_Double_Bidirectional_GRU_backward_rnn_25_while_cond_387378459
tree_encoder_double_bidirectional_gru_backward_rnn_25_while_tree_encoder_double_bidirectional_gru_backward_rnn_25_while_loop_counter
tree_encoder_double_bidirectional_gru_backward_rnn_25_while_tree_encoder_double_bidirectional_gru_backward_rnn_25_while_maximum_iterationsK
Gtree_encoder_double_bidirectional_gru_backward_rnn_25_while_placeholderM
Itree_encoder_double_bidirectional_gru_backward_rnn_25_while_placeholder_1M
Itree_encoder_double_bidirectional_gru_backward_rnn_25_while_placeholder_2
tree_encoder_double_bidirectional_gru_backward_rnn_25_while_less_tree_encoder_double_bidirectional_gru_backward_rnn_25_strided_slice_1¤
tree_encoder_double_bidirectional_gru_backward_rnn_25_while_tree_encoder_double_bidirectional_gru_backward_rnn_25_while_cond_387378459___redundant_placeholder0¤
tree_encoder_double_bidirectional_gru_backward_rnn_25_while_tree_encoder_double_bidirectional_gru_backward_rnn_25_while_cond_387378459___redundant_placeholder1¤
tree_encoder_double_bidirectional_gru_backward_rnn_25_while_tree_encoder_double_bidirectional_gru_backward_rnn_25_while_cond_387378459___redundant_placeholder2¤
tree_encoder_double_bidirectional_gru_backward_rnn_25_while_tree_encoder_double_bidirectional_gru_backward_rnn_25_while_cond_387378459___redundant_placeholder3H
Dtree_encoder_double_bidirectional_gru_backward_rnn_25_while_identity
ÿ
@Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/LessLessGtree_encoder_double_bidirectional_gru_backward_rnn_25_while_placeholdertree_encoder_double_bidirectional_gru_backward_rnn_25_while_less_tree_encoder_double_bidirectional_gru_backward_rnn_25_strided_slice_1*
T0*
_output_shapes
: 2B
@Tree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/Lessÿ
DTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/IdentityIdentityDTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/Less:z:0*
T0
*
_output_shapes
: 2F
DTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/Identity"
Dtree_encoder_double_bidirectional_gru_backward_rnn_25_while_identityMTree_Encoder/Double_Bidirectional_GRU/backward_rnn_25/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Â

à
/__inference_gru_cell_25_layer_call_fn_387382833

inputs
states_0
unknown:	
	unknown_0:

	unknown_1:

identity

identity_1¢StatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_gru_cell_25_layer_call_and_return_conditional_losses_3873793532
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0
Â

à
/__inference_gru_cell_25_layer_call_fn_387382819

inputs
states_0
unknown:	
	unknown_0:

	unknown_1:

identity

identity_1¢StatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_gru_cell_25_layer_call_and_return_conditional_losses_3873792112
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0
Äa
ï
$backward_rnn_25_while_body_387379957<
8backward_rnn_25_while_backward_rnn_25_while_loop_counterB
>backward_rnn_25_while_backward_rnn_25_while_maximum_iterations%
!backward_rnn_25_while_placeholder'
#backward_rnn_25_while_placeholder_1'
#backward_rnn_25_while_placeholder_2;
7backward_rnn_25_while_backward_rnn_25_strided_slice_1_0w
sbackward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_backward_rnn_25_tensorarrayunstack_tensorlistfromtensor_0N
;backward_rnn_25_while_gru_cell_25_readvariableop_resource_0:	V
Bbackward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource_0:
X
Dbackward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource_0:
"
backward_rnn_25_while_identity$
 backward_rnn_25_while_identity_1$
 backward_rnn_25_while_identity_2$
 backward_rnn_25_while_identity_3$
 backward_rnn_25_while_identity_49
5backward_rnn_25_while_backward_rnn_25_strided_slice_1u
qbackward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_backward_rnn_25_tensorarrayunstack_tensorlistfromtensorL
9backward_rnn_25_while_gru_cell_25_readvariableop_resource:	T
@backward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource:
V
Bbackward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource:
¢7backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp¢9backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp¢0backward_rnn_25/while/gru_cell_25/ReadVariableOpã
Gbackward_rnn_25/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2I
Gbackward_rnn_25/while/TensorArrayV2Read/TensorListGetItem/element_shape´
9backward_rnn_25/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsbackward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_backward_rnn_25_tensorarrayunstack_tensorlistfromtensor_0!backward_rnn_25_while_placeholderPbackward_rnn_25/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02;
9backward_rnn_25/while/TensorArrayV2Read/TensorListGetItemá
0backward_rnn_25/while/gru_cell_25/ReadVariableOpReadVariableOp;backward_rnn_25_while_gru_cell_25_readvariableop_resource_0*
_output_shapes
:	*
dtype022
0backward_rnn_25/while/gru_cell_25/ReadVariableOpÒ
)backward_rnn_25/while/gru_cell_25/unstackUnpack8backward_rnn_25/while/gru_cell_25/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2+
)backward_rnn_25/while/gru_cell_25/unstack÷
7backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOpReadVariableOpBbackward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype029
7backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp
(backward_rnn_25/while/gru_cell_25/MatMulMatMul@backward_rnn_25/while/TensorArrayV2Read/TensorListGetItem:item:0?backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(backward_rnn_25/while/gru_cell_25/MatMulü
)backward_rnn_25/while/gru_cell_25/BiasAddBiasAdd2backward_rnn_25/while/gru_cell_25/MatMul:product:02backward_rnn_25/while/gru_cell_25/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)backward_rnn_25/while/gru_cell_25/BiasAdd±
1backward_rnn_25/while/gru_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ23
1backward_rnn_25/while/gru_cell_25/split/split_dim·
'backward_rnn_25/while/gru_cell_25/splitSplit:backward_rnn_25/while/gru_cell_25/split/split_dim:output:02backward_rnn_25/while/gru_cell_25/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2)
'backward_rnn_25/while/gru_cell_25/splitý
9backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpReadVariableOpDbackward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02;
9backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpý
*backward_rnn_25/while/gru_cell_25/MatMul_1MatMul#backward_rnn_25_while_placeholder_2Abackward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*backward_rnn_25/while/gru_cell_25/MatMul_1
+backward_rnn_25/while/gru_cell_25/BiasAdd_1BiasAdd4backward_rnn_25/while/gru_cell_25/MatMul_1:product:02backward_rnn_25/while/gru_cell_25/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+backward_rnn_25/while/gru_cell_25/BiasAdd_1§
'backward_rnn_25/while/gru_cell_25/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2)
'backward_rnn_25/while/gru_cell_25/Constµ
3backward_rnn_25/while/gru_cell_25/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ25
3backward_rnn_25/while/gru_cell_25/split_1/split_dimþ
)backward_rnn_25/while/gru_cell_25/split_1SplitV4backward_rnn_25/while/gru_cell_25/BiasAdd_1:output:00backward_rnn_25/while/gru_cell_25/Const:output:0<backward_rnn_25/while/gru_cell_25/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2+
)backward_rnn_25/while/gru_cell_25/split_1ð
%backward_rnn_25/while/gru_cell_25/addAddV20backward_rnn_25/while/gru_cell_25/split:output:02backward_rnn_25/while/gru_cell_25/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%backward_rnn_25/while/gru_cell_25/add¿
)backward_rnn_25/while/gru_cell_25/SigmoidSigmoid)backward_rnn_25/while/gru_cell_25/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)backward_rnn_25/while/gru_cell_25/Sigmoidô
'backward_rnn_25/while/gru_cell_25/add_1AddV20backward_rnn_25/while/gru_cell_25/split:output:12backward_rnn_25/while/gru_cell_25/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'backward_rnn_25/while/gru_cell_25/add_1Å
+backward_rnn_25/while/gru_cell_25/Sigmoid_1Sigmoid+backward_rnn_25/while/gru_cell_25/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+backward_rnn_25/while/gru_cell_25/Sigmoid_1í
%backward_rnn_25/while/gru_cell_25/mulMul/backward_rnn_25/while/gru_cell_25/Sigmoid_1:y:02backward_rnn_25/while/gru_cell_25/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%backward_rnn_25/while/gru_cell_25/mulë
'backward_rnn_25/while/gru_cell_25/add_2AddV20backward_rnn_25/while/gru_cell_25/split:output:2)backward_rnn_25/while/gru_cell_25/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'backward_rnn_25/while/gru_cell_25/add_2¸
&backward_rnn_25/while/gru_cell_25/TanhTanh+backward_rnn_25/while/gru_cell_25/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&backward_rnn_25/while/gru_cell_25/Tanhà
'backward_rnn_25/while/gru_cell_25/mul_1Mul-backward_rnn_25/while/gru_cell_25/Sigmoid:y:0#backward_rnn_25_while_placeholder_2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'backward_rnn_25/while/gru_cell_25/mul_1
'backward_rnn_25/while/gru_cell_25/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'backward_rnn_25/while/gru_cell_25/sub/xé
%backward_rnn_25/while/gru_cell_25/subSub0backward_rnn_25/while/gru_cell_25/sub/x:output:0-backward_rnn_25/while/gru_cell_25/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%backward_rnn_25/while/gru_cell_25/subã
'backward_rnn_25/while/gru_cell_25/mul_2Mul)backward_rnn_25/while/gru_cell_25/sub:z:0*backward_rnn_25/while/gru_cell_25/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'backward_rnn_25/while/gru_cell_25/mul_2è
'backward_rnn_25/while/gru_cell_25/add_3AddV2+backward_rnn_25/while/gru_cell_25/mul_1:z:0+backward_rnn_25/while/gru_cell_25/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'backward_rnn_25/while/gru_cell_25/add_3¯
:backward_rnn_25/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#backward_rnn_25_while_placeholder_1!backward_rnn_25_while_placeholder+backward_rnn_25/while/gru_cell_25/add_3:z:0*
_output_shapes
: *
element_dtype02<
:backward_rnn_25/while/TensorArrayV2Write/TensorListSetItem|
backward_rnn_25/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_rnn_25/while/add/y©
backward_rnn_25/while/addAddV2!backward_rnn_25_while_placeholder$backward_rnn_25/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_rnn_25/while/add
backward_rnn_25/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_rnn_25/while/add_1/yÆ
backward_rnn_25/while/add_1AddV28backward_rnn_25_while_backward_rnn_25_while_loop_counter&backward_rnn_25/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_rnn_25/while/add_1·
backward_rnn_25/while/IdentityIdentitybackward_rnn_25/while/add_1:z:08^backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp:^backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp1^backward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2 
backward_rnn_25/while/IdentityÚ
 backward_rnn_25/while/Identity_1Identity>backward_rnn_25_while_backward_rnn_25_while_maximum_iterations8^backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp:^backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp1^backward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2"
 backward_rnn_25/while/Identity_1¹
 backward_rnn_25/while/Identity_2Identitybackward_rnn_25/while/add:z:08^backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp:^backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp1^backward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2"
 backward_rnn_25/while/Identity_2æ
 backward_rnn_25/while/Identity_3IdentityJbackward_rnn_25/while/TensorArrayV2Write/TensorListSetItem:output_handle:08^backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp:^backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp1^backward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2"
 backward_rnn_25/while/Identity_3Ù
 backward_rnn_25/while/Identity_4Identity+backward_rnn_25/while/gru_cell_25/add_3:z:08^backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp:^backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp1^backward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 backward_rnn_25/while/Identity_4"p
5backward_rnn_25_while_backward_rnn_25_strided_slice_17backward_rnn_25_while_backward_rnn_25_strided_slice_1_0"
Bbackward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resourceDbackward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource_0"
@backward_rnn_25_while_gru_cell_25_matmul_readvariableop_resourceBbackward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource_0"x
9backward_rnn_25_while_gru_cell_25_readvariableop_resource;backward_rnn_25_while_gru_cell_25_readvariableop_resource_0"I
backward_rnn_25_while_identity'backward_rnn_25/while/Identity:output:0"M
 backward_rnn_25_while_identity_1)backward_rnn_25/while/Identity_1:output:0"M
 backward_rnn_25_while_identity_2)backward_rnn_25/while/Identity_2:output:0"M
 backward_rnn_25_while_identity_3)backward_rnn_25/while/Identity_3:output:0"M
 backward_rnn_25_while_identity_4)backward_rnn_25/while/Identity_4:output:0"è
qbackward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_backward_rnn_25_tensorarrayunstack_tensorlistfromtensorsbackward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_backward_rnn_25_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : 2r
7backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp7backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp2v
9backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp9backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp2d
0backward_rnn_25/while/gru_cell_25/ReadVariableOp0backward_rnn_25/while/gru_cell_25/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
·

û
I__inference_Prediction_layer_call_and_return_conditional_losses_387380072

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
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
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
×
Þ
J__inference_gru_cell_25_layer_call_and_return_conditional_losses_387378773

inputs

states*
readvariableop_resource:	2
matmul_readvariableop_resource:
4
 matmul_1_readvariableop_resource:

identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpy
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
split/split_dim¯
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	BiasAdd_1c
ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2
Constq
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
split_1/split_dimÔ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanh]
mul_1MulSigmoid:y:0states*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sub[
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_3¡
IdentityIdentity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¥

Identity_1Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
×
Þ
J__inference_gru_cell_25_layer_call_and_return_conditional_losses_387378631

inputs

states*
readvariableop_resource:	2
matmul_readvariableop_resource:
4
 matmul_1_readvariableop_resource:

identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpy
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
split/split_dim¯
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	BiasAdd_1c
ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2
Constq
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
split_1/split_dimÔ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanh]
mul_1MulSigmoid:y:0states*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sub[
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_3¡
IdentityIdentity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¥

Identity_1Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
À
Ö
#forward_rnn_25_while_cond_387381762:
6forward_rnn_25_while_forward_rnn_25_while_loop_counter@
<forward_rnn_25_while_forward_rnn_25_while_maximum_iterations$
 forward_rnn_25_while_placeholder&
"forward_rnn_25_while_placeholder_1&
"forward_rnn_25_while_placeholder_2<
8forward_rnn_25_while_less_forward_rnn_25_strided_slice_1U
Qforward_rnn_25_while_forward_rnn_25_while_cond_387381762___redundant_placeholder0U
Qforward_rnn_25_while_forward_rnn_25_while_cond_387381762___redundant_placeholder1U
Qforward_rnn_25_while_forward_rnn_25_while_cond_387381762___redundant_placeholder2U
Qforward_rnn_25_while_forward_rnn_25_while_cond_387381762___redundant_placeholder3!
forward_rnn_25_while_identity
»
forward_rnn_25/while/LessLess forward_rnn_25_while_placeholder8forward_rnn_25_while_less_forward_rnn_25_strided_slice_1*
T0*
_output_shapes
: 2
forward_rnn_25/while/Less
forward_rnn_25/while/IdentityIdentityforward_rnn_25/while/Less:z:0*
T0
*
_output_shapes
: 2
forward_rnn_25/while/Identity"G
forward_rnn_25_while_identity&forward_rnn_25/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
º
Ì	
=Double_Bidirectional_GRU_backward_rnn_25_while_cond_387380897n
jdouble_bidirectional_gru_backward_rnn_25_while_double_bidirectional_gru_backward_rnn_25_while_loop_countert
pdouble_bidirectional_gru_backward_rnn_25_while_double_bidirectional_gru_backward_rnn_25_while_maximum_iterations>
:double_bidirectional_gru_backward_rnn_25_while_placeholder@
<double_bidirectional_gru_backward_rnn_25_while_placeholder_1@
<double_bidirectional_gru_backward_rnn_25_while_placeholder_2p
ldouble_bidirectional_gru_backward_rnn_25_while_less_double_bidirectional_gru_backward_rnn_25_strided_slice_1
double_bidirectional_gru_backward_rnn_25_while_double_bidirectional_gru_backward_rnn_25_while_cond_387380897___redundant_placeholder0
double_bidirectional_gru_backward_rnn_25_while_double_bidirectional_gru_backward_rnn_25_while_cond_387380897___redundant_placeholder1
double_bidirectional_gru_backward_rnn_25_while_double_bidirectional_gru_backward_rnn_25_while_cond_387380897___redundant_placeholder2
double_bidirectional_gru_backward_rnn_25_while_double_bidirectional_gru_backward_rnn_25_while_cond_387380897___redundant_placeholder3;
7double_bidirectional_gru_backward_rnn_25_while_identity
½
3Double_Bidirectional_GRU/backward_rnn_25/while/LessLess:double_bidirectional_gru_backward_rnn_25_while_placeholderldouble_bidirectional_gru_backward_rnn_25_while_less_double_bidirectional_gru_backward_rnn_25_strided_slice_1*
T0*
_output_shapes
: 25
3Double_Bidirectional_GRU/backward_rnn_25/while/LessØ
7Double_Bidirectional_GRU/backward_rnn_25/while/IdentityIdentity7Double_Bidirectional_GRU/backward_rnn_25/while/Less:z:0*
T0
*
_output_shapes
: 29
7Double_Bidirectional_GRU/backward_rnn_25/while/Identity"{
7double_bidirectional_gru_backward_rnn_25_while_identity@Double_Bidirectional_GRU/backward_rnn_25/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Äa
ï
$backward_rnn_25_while_body_387381605<
8backward_rnn_25_while_backward_rnn_25_while_loop_counterB
>backward_rnn_25_while_backward_rnn_25_while_maximum_iterations%
!backward_rnn_25_while_placeholder'
#backward_rnn_25_while_placeholder_1'
#backward_rnn_25_while_placeholder_2;
7backward_rnn_25_while_backward_rnn_25_strided_slice_1_0w
sbackward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_backward_rnn_25_tensorarrayunstack_tensorlistfromtensor_0N
;backward_rnn_25_while_gru_cell_25_readvariableop_resource_0:	V
Bbackward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource_0:
X
Dbackward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource_0:
"
backward_rnn_25_while_identity$
 backward_rnn_25_while_identity_1$
 backward_rnn_25_while_identity_2$
 backward_rnn_25_while_identity_3$
 backward_rnn_25_while_identity_49
5backward_rnn_25_while_backward_rnn_25_strided_slice_1u
qbackward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_backward_rnn_25_tensorarrayunstack_tensorlistfromtensorL
9backward_rnn_25_while_gru_cell_25_readvariableop_resource:	T
@backward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource:
V
Bbackward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource:
¢7backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp¢9backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp¢0backward_rnn_25/while/gru_cell_25/ReadVariableOpã
Gbackward_rnn_25/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2I
Gbackward_rnn_25/while/TensorArrayV2Read/TensorListGetItem/element_shape´
9backward_rnn_25/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsbackward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_backward_rnn_25_tensorarrayunstack_tensorlistfromtensor_0!backward_rnn_25_while_placeholderPbackward_rnn_25/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02;
9backward_rnn_25/while/TensorArrayV2Read/TensorListGetItemá
0backward_rnn_25/while/gru_cell_25/ReadVariableOpReadVariableOp;backward_rnn_25_while_gru_cell_25_readvariableop_resource_0*
_output_shapes
:	*
dtype022
0backward_rnn_25/while/gru_cell_25/ReadVariableOpÒ
)backward_rnn_25/while/gru_cell_25/unstackUnpack8backward_rnn_25/while/gru_cell_25/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2+
)backward_rnn_25/while/gru_cell_25/unstack÷
7backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOpReadVariableOpBbackward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype029
7backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp
(backward_rnn_25/while/gru_cell_25/MatMulMatMul@backward_rnn_25/while/TensorArrayV2Read/TensorListGetItem:item:0?backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(backward_rnn_25/while/gru_cell_25/MatMulü
)backward_rnn_25/while/gru_cell_25/BiasAddBiasAdd2backward_rnn_25/while/gru_cell_25/MatMul:product:02backward_rnn_25/while/gru_cell_25/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)backward_rnn_25/while/gru_cell_25/BiasAdd±
1backward_rnn_25/while/gru_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ23
1backward_rnn_25/while/gru_cell_25/split/split_dim·
'backward_rnn_25/while/gru_cell_25/splitSplit:backward_rnn_25/while/gru_cell_25/split/split_dim:output:02backward_rnn_25/while/gru_cell_25/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2)
'backward_rnn_25/while/gru_cell_25/splitý
9backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpReadVariableOpDbackward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02;
9backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpý
*backward_rnn_25/while/gru_cell_25/MatMul_1MatMul#backward_rnn_25_while_placeholder_2Abackward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*backward_rnn_25/while/gru_cell_25/MatMul_1
+backward_rnn_25/while/gru_cell_25/BiasAdd_1BiasAdd4backward_rnn_25/while/gru_cell_25/MatMul_1:product:02backward_rnn_25/while/gru_cell_25/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+backward_rnn_25/while/gru_cell_25/BiasAdd_1§
'backward_rnn_25/while/gru_cell_25/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2)
'backward_rnn_25/while/gru_cell_25/Constµ
3backward_rnn_25/while/gru_cell_25/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ25
3backward_rnn_25/while/gru_cell_25/split_1/split_dimþ
)backward_rnn_25/while/gru_cell_25/split_1SplitV4backward_rnn_25/while/gru_cell_25/BiasAdd_1:output:00backward_rnn_25/while/gru_cell_25/Const:output:0<backward_rnn_25/while/gru_cell_25/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2+
)backward_rnn_25/while/gru_cell_25/split_1ð
%backward_rnn_25/while/gru_cell_25/addAddV20backward_rnn_25/while/gru_cell_25/split:output:02backward_rnn_25/while/gru_cell_25/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%backward_rnn_25/while/gru_cell_25/add¿
)backward_rnn_25/while/gru_cell_25/SigmoidSigmoid)backward_rnn_25/while/gru_cell_25/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)backward_rnn_25/while/gru_cell_25/Sigmoidô
'backward_rnn_25/while/gru_cell_25/add_1AddV20backward_rnn_25/while/gru_cell_25/split:output:12backward_rnn_25/while/gru_cell_25/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'backward_rnn_25/while/gru_cell_25/add_1Å
+backward_rnn_25/while/gru_cell_25/Sigmoid_1Sigmoid+backward_rnn_25/while/gru_cell_25/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+backward_rnn_25/while/gru_cell_25/Sigmoid_1í
%backward_rnn_25/while/gru_cell_25/mulMul/backward_rnn_25/while/gru_cell_25/Sigmoid_1:y:02backward_rnn_25/while/gru_cell_25/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%backward_rnn_25/while/gru_cell_25/mulë
'backward_rnn_25/while/gru_cell_25/add_2AddV20backward_rnn_25/while/gru_cell_25/split:output:2)backward_rnn_25/while/gru_cell_25/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'backward_rnn_25/while/gru_cell_25/add_2¸
&backward_rnn_25/while/gru_cell_25/TanhTanh+backward_rnn_25/while/gru_cell_25/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&backward_rnn_25/while/gru_cell_25/Tanhà
'backward_rnn_25/while/gru_cell_25/mul_1Mul-backward_rnn_25/while/gru_cell_25/Sigmoid:y:0#backward_rnn_25_while_placeholder_2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'backward_rnn_25/while/gru_cell_25/mul_1
'backward_rnn_25/while/gru_cell_25/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'backward_rnn_25/while/gru_cell_25/sub/xé
%backward_rnn_25/while/gru_cell_25/subSub0backward_rnn_25/while/gru_cell_25/sub/x:output:0-backward_rnn_25/while/gru_cell_25/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%backward_rnn_25/while/gru_cell_25/subã
'backward_rnn_25/while/gru_cell_25/mul_2Mul)backward_rnn_25/while/gru_cell_25/sub:z:0*backward_rnn_25/while/gru_cell_25/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'backward_rnn_25/while/gru_cell_25/mul_2è
'backward_rnn_25/while/gru_cell_25/add_3AddV2+backward_rnn_25/while/gru_cell_25/mul_1:z:0+backward_rnn_25/while/gru_cell_25/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'backward_rnn_25/while/gru_cell_25/add_3¯
:backward_rnn_25/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#backward_rnn_25_while_placeholder_1!backward_rnn_25_while_placeholder+backward_rnn_25/while/gru_cell_25/add_3:z:0*
_output_shapes
: *
element_dtype02<
:backward_rnn_25/while/TensorArrayV2Write/TensorListSetItem|
backward_rnn_25/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_rnn_25/while/add/y©
backward_rnn_25/while/addAddV2!backward_rnn_25_while_placeholder$backward_rnn_25/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_rnn_25/while/add
backward_rnn_25/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_rnn_25/while/add_1/yÆ
backward_rnn_25/while/add_1AddV28backward_rnn_25_while_backward_rnn_25_while_loop_counter&backward_rnn_25/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_rnn_25/while/add_1·
backward_rnn_25/while/IdentityIdentitybackward_rnn_25/while/add_1:z:08^backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp:^backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp1^backward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2 
backward_rnn_25/while/IdentityÚ
 backward_rnn_25/while/Identity_1Identity>backward_rnn_25_while_backward_rnn_25_while_maximum_iterations8^backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp:^backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp1^backward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2"
 backward_rnn_25/while/Identity_1¹
 backward_rnn_25/while/Identity_2Identitybackward_rnn_25/while/add:z:08^backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp:^backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp1^backward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2"
 backward_rnn_25/while/Identity_2æ
 backward_rnn_25/while/Identity_3IdentityJbackward_rnn_25/while/TensorArrayV2Write/TensorListSetItem:output_handle:08^backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp:^backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp1^backward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2"
 backward_rnn_25/while/Identity_3Ù
 backward_rnn_25/while/Identity_4Identity+backward_rnn_25/while/gru_cell_25/add_3:z:08^backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp:^backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp1^backward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 backward_rnn_25/while/Identity_4"p
5backward_rnn_25_while_backward_rnn_25_strided_slice_17backward_rnn_25_while_backward_rnn_25_strided_slice_1_0"
Bbackward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resourceDbackward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource_0"
@backward_rnn_25_while_gru_cell_25_matmul_readvariableop_resourceBbackward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource_0"x
9backward_rnn_25_while_gru_cell_25_readvariableop_resource;backward_rnn_25_while_gru_cell_25_readvariableop_resource_0"I
backward_rnn_25_while_identity'backward_rnn_25/while/Identity:output:0"M
 backward_rnn_25_while_identity_1)backward_rnn_25/while/Identity_1:output:0"M
 backward_rnn_25_while_identity_2)backward_rnn_25/while/Identity_2:output:0"M
 backward_rnn_25_while_identity_3)backward_rnn_25/while/Identity_3:output:0"M
 backward_rnn_25_while_identity_4)backward_rnn_25/while/Identity_4:output:0"è
qbackward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_backward_rnn_25_tensorarrayunstack_tensorlistfromtensorsbackward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_backward_rnn_25_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : 2r
7backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp7backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp2v
9backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp9backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp2d
0backward_rnn_25/while/gru_cell_25/ReadVariableOp0backward_rnn_25/while/gru_cell_25/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
«

.__inference_Prediction_layer_call_fn_387382016

inputs
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallü
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
GPU2*0J 8 *R
fMRK
I__inference_Prediction_layer_call_and_return_conditional_losses_3873800722
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
H

while_body_387382273
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_25_readvariableop_resource_0:	F
2while_gru_cell_25_matmul_readvariableop_resource_0:
H
4while_gru_cell_25_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_25_readvariableop_resource:	D
0while_gru_cell_25_matmul_readvariableop_resource:
F
2while_gru_cell_25_matmul_1_readvariableop_resource:
¢'while/gru_cell_25/MatMul/ReadVariableOp¢)while/gru_cell_25/MatMul_1/ReadVariableOp¢ while/gru_cell_25/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem±
 while/gru_cell_25/ReadVariableOpReadVariableOp+while_gru_cell_25_readvariableop_resource_0*
_output_shapes
:	*
dtype02"
 while/gru_cell_25/ReadVariableOp¢
while/gru_cell_25/unstackUnpack(while/gru_cell_25/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
while/gru_cell_25/unstackÇ
'while/gru_cell_25/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_25_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype02)
'while/gru_cell_25/MatMul/ReadVariableOpÔ
while/gru_cell_25/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/MatMul¼
while/gru_cell_25/BiasAddBiasAdd"while/gru_cell_25/MatMul:product:0"while/gru_cell_25/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/BiasAdd
!while/gru_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2#
!while/gru_cell_25/split/split_dim÷
while/gru_cell_25/splitSplit*while/gru_cell_25/split/split_dim:output:0"while/gru_cell_25/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/gru_cell_25/splitÍ
)while/gru_cell_25/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_25_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02+
)while/gru_cell_25/MatMul_1/ReadVariableOp½
while/gru_cell_25/MatMul_1MatMulwhile_placeholder_21while/gru_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/MatMul_1Â
while/gru_cell_25/BiasAdd_1BiasAdd$while/gru_cell_25/MatMul_1:product:0"while/gru_cell_25/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/BiasAdd_1
while/gru_cell_25/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2
while/gru_cell_25/Const
#while/gru_cell_25/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#while/gru_cell_25/split_1/split_dim®
while/gru_cell_25/split_1SplitV$while/gru_cell_25/BiasAdd_1:output:0 while/gru_cell_25/Const:output:0,while/gru_cell_25/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/gru_cell_25/split_1°
while/gru_cell_25/addAddV2 while/gru_cell_25/split:output:0"while/gru_cell_25/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/add
while/gru_cell_25/SigmoidSigmoidwhile/gru_cell_25/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/Sigmoid´
while/gru_cell_25/add_1AddV2 while/gru_cell_25/split:output:1"while/gru_cell_25/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/add_1
while/gru_cell_25/Sigmoid_1Sigmoidwhile/gru_cell_25/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/Sigmoid_1­
while/gru_cell_25/mulMulwhile/gru_cell_25/Sigmoid_1:y:0"while/gru_cell_25/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/mul«
while/gru_cell_25/add_2AddV2 while/gru_cell_25/split:output:2while/gru_cell_25/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/add_2
while/gru_cell_25/TanhTanhwhile/gru_cell_25/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/Tanh 
while/gru_cell_25/mul_1Mulwhile/gru_cell_25/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/mul_1w
while/gru_cell_25/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_25/sub/x©
while/gru_cell_25/subSub while/gru_cell_25/sub/x:output:0while/gru_cell_25/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/sub£
while/gru_cell_25/mul_2Mulwhile/gru_cell_25/sub:z:0while/gru_cell_25/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/mul_2¨
while/gru_cell_25/add_3AddV2while/gru_cell_25/mul_1:z:0while/gru_cell_25/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/add_3ß
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_25/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1×
while/IdentityIdentitywhile/add_1:z:0(^while/gru_cell_25/MatMul/ReadVariableOp*^while/gru_cell_25/MatMul_1/ReadVariableOp!^while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityê
while/Identity_1Identitywhile_while_maximum_iterations(^while/gru_cell_25/MatMul/ReadVariableOp*^while/gru_cell_25/MatMul_1/ReadVariableOp!^while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1Ù
while/Identity_2Identitywhile/add:z:0(^while/gru_cell_25/MatMul/ReadVariableOp*^while/gru_cell_25/MatMul_1/ReadVariableOp!^while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/gru_cell_25/MatMul/ReadVariableOp*^while/gru_cell_25/MatMul_1/ReadVariableOp!^while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3ù
while/Identity_4Identitywhile/gru_cell_25/add_3:z:0(^while/gru_cell_25/MatMul/ReadVariableOp*^while/gru_cell_25/MatMul_1/ReadVariableOp!^while/gru_cell_25/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4"j
2while_gru_cell_25_matmul_1_readvariableop_resource4while_gru_cell_25_matmul_1_readvariableop_resource_0"f
0while_gru_cell_25_matmul_readvariableop_resource2while_gru_cell_25_matmul_readvariableop_resource_0"X
)while_gru_cell_25_readvariableop_resource+while_gru_cell_25_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : 2R
'while/gru_cell_25/MatMul/ReadVariableOp'while/gru_cell_25/MatMul/ReadVariableOp2V
)while/gru_cell_25/MatMul_1/ReadVariableOp)while/gru_cell_25/MatMul_1/ReadVariableOp2D
 while/gru_cell_25/ReadVariableOp while/gru_cell_25/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ñ
¡
W__inference_Double_Bidirectional_GRU_layer_call_and_return_conditional_losses_387381695

inputsE
2forward_rnn_25_gru_cell_25_readvariableop_resource:	M
9forward_rnn_25_gru_cell_25_matmul_readvariableop_resource:
O
;forward_rnn_25_gru_cell_25_matmul_1_readvariableop_resource:
F
3backward_rnn_25_gru_cell_25_readvariableop_resource:	N
:backward_rnn_25_gru_cell_25_matmul_readvariableop_resource:
P
<backward_rnn_25_gru_cell_25_matmul_1_readvariableop_resource:

identity¢1backward_rnn_25/gru_cell_25/MatMul/ReadVariableOp¢3backward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp¢*backward_rnn_25/gru_cell_25/ReadVariableOp¢backward_rnn_25/while¢0forward_rnn_25/gru_cell_25/MatMul/ReadVariableOp¢2forward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp¢)forward_rnn_25/gru_cell_25/ReadVariableOp¢forward_rnn_25/whileb
forward_rnn_25/ShapeShapeinputs*
T0*
_output_shapes
:2
forward_rnn_25/Shape
"forward_rnn_25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"forward_rnn_25/strided_slice/stack
$forward_rnn_25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$forward_rnn_25/strided_slice/stack_1
$forward_rnn_25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$forward_rnn_25/strided_slice/stack_2¼
forward_rnn_25/strided_sliceStridedSliceforward_rnn_25/Shape:output:0+forward_rnn_25/strided_slice/stack:output:0-forward_rnn_25/strided_slice/stack_1:output:0-forward_rnn_25/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
forward_rnn_25/strided_slice{
forward_rnn_25/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
forward_rnn_25/zeros/mul/y¨
forward_rnn_25/zeros/mulMul%forward_rnn_25/strided_slice:output:0#forward_rnn_25/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
forward_rnn_25/zeros/mul}
forward_rnn_25/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
forward_rnn_25/zeros/Less/y£
forward_rnn_25/zeros/LessLessforward_rnn_25/zeros/mul:z:0$forward_rnn_25/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
forward_rnn_25/zeros/Less
forward_rnn_25/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
forward_rnn_25/zeros/packed/1¿
forward_rnn_25/zeros/packedPack%forward_rnn_25/strided_slice:output:0&forward_rnn_25/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
forward_rnn_25/zeros/packed}
forward_rnn_25/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
forward_rnn_25/zeros/Const²
forward_rnn_25/zerosFill$forward_rnn_25/zeros/packed:output:0#forward_rnn_25/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
forward_rnn_25/zeros
forward_rnn_25/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
forward_rnn_25/transpose/perm±
forward_rnn_25/transpose	Transposeinputs&forward_rnn_25/transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
forward_rnn_25/transpose|
forward_rnn_25/Shape_1Shapeforward_rnn_25/transpose:y:0*
T0*
_output_shapes
:2
forward_rnn_25/Shape_1
$forward_rnn_25/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_rnn_25/strided_slice_1/stack
&forward_rnn_25/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_rnn_25/strided_slice_1/stack_1
&forward_rnn_25/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_rnn_25/strided_slice_1/stack_2È
forward_rnn_25/strided_slice_1StridedSliceforward_rnn_25/Shape_1:output:0-forward_rnn_25/strided_slice_1/stack:output:0/forward_rnn_25/strided_slice_1/stack_1:output:0/forward_rnn_25/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
forward_rnn_25/strided_slice_1£
*forward_rnn_25/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2,
*forward_rnn_25/TensorArrayV2/element_shapeî
forward_rnn_25/TensorArrayV2TensorListReserve3forward_rnn_25/TensorArrayV2/element_shape:output:0'forward_rnn_25/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
forward_rnn_25/TensorArrayV2Ý
Dforward_rnn_25/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2F
Dforward_rnn_25/TensorArrayUnstack/TensorListFromTensor/element_shape´
6forward_rnn_25/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_rnn_25/transpose:y:0Mforward_rnn_25/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type028
6forward_rnn_25/TensorArrayUnstack/TensorListFromTensor
$forward_rnn_25/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$forward_rnn_25/strided_slice_2/stack
&forward_rnn_25/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_rnn_25/strided_slice_2/stack_1
&forward_rnn_25/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_rnn_25/strided_slice_2/stack_2×
forward_rnn_25/strided_slice_2StridedSliceforward_rnn_25/transpose:y:0-forward_rnn_25/strided_slice_2/stack:output:0/forward_rnn_25/strided_slice_2/stack_1:output:0/forward_rnn_25/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2 
forward_rnn_25/strided_slice_2Ê
)forward_rnn_25/gru_cell_25/ReadVariableOpReadVariableOp2forward_rnn_25_gru_cell_25_readvariableop_resource*
_output_shapes
:	*
dtype02+
)forward_rnn_25/gru_cell_25/ReadVariableOp½
"forward_rnn_25/gru_cell_25/unstackUnpack1forward_rnn_25/gru_cell_25/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2$
"forward_rnn_25/gru_cell_25/unstackà
0forward_rnn_25/gru_cell_25/MatMul/ReadVariableOpReadVariableOp9forward_rnn_25_gru_cell_25_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype022
0forward_rnn_25/gru_cell_25/MatMul/ReadVariableOpæ
!forward_rnn_25/gru_cell_25/MatMulMatMul'forward_rnn_25/strided_slice_2:output:08forward_rnn_25/gru_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!forward_rnn_25/gru_cell_25/MatMulà
"forward_rnn_25/gru_cell_25/BiasAddBiasAdd+forward_rnn_25/gru_cell_25/MatMul:product:0+forward_rnn_25/gru_cell_25/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"forward_rnn_25/gru_cell_25/BiasAdd£
*forward_rnn_25/gru_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2,
*forward_rnn_25/gru_cell_25/split/split_dim
 forward_rnn_25/gru_cell_25/splitSplit3forward_rnn_25/gru_cell_25/split/split_dim:output:0+forward_rnn_25/gru_cell_25/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2"
 forward_rnn_25/gru_cell_25/splitæ
2forward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOpReadVariableOp;forward_rnn_25_gru_cell_25_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype024
2forward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOpâ
#forward_rnn_25/gru_cell_25/MatMul_1MatMulforward_rnn_25/zeros:output:0:forward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#forward_rnn_25/gru_cell_25/MatMul_1æ
$forward_rnn_25/gru_cell_25/BiasAdd_1BiasAdd-forward_rnn_25/gru_cell_25/MatMul_1:product:0+forward_rnn_25/gru_cell_25/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$forward_rnn_25/gru_cell_25/BiasAdd_1
 forward_rnn_25/gru_cell_25/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2"
 forward_rnn_25/gru_cell_25/Const§
,forward_rnn_25/gru_cell_25/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2.
,forward_rnn_25/gru_cell_25/split_1/split_dimÛ
"forward_rnn_25/gru_cell_25/split_1SplitV-forward_rnn_25/gru_cell_25/BiasAdd_1:output:0)forward_rnn_25/gru_cell_25/Const:output:05forward_rnn_25/gru_cell_25/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2$
"forward_rnn_25/gru_cell_25/split_1Ô
forward_rnn_25/gru_cell_25/addAddV2)forward_rnn_25/gru_cell_25/split:output:0+forward_rnn_25/gru_cell_25/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
forward_rnn_25/gru_cell_25/addª
"forward_rnn_25/gru_cell_25/SigmoidSigmoid"forward_rnn_25/gru_cell_25/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"forward_rnn_25/gru_cell_25/SigmoidØ
 forward_rnn_25/gru_cell_25/add_1AddV2)forward_rnn_25/gru_cell_25/split:output:1+forward_rnn_25/gru_cell_25/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 forward_rnn_25/gru_cell_25/add_1°
$forward_rnn_25/gru_cell_25/Sigmoid_1Sigmoid$forward_rnn_25/gru_cell_25/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$forward_rnn_25/gru_cell_25/Sigmoid_1Ñ
forward_rnn_25/gru_cell_25/mulMul(forward_rnn_25/gru_cell_25/Sigmoid_1:y:0+forward_rnn_25/gru_cell_25/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
forward_rnn_25/gru_cell_25/mulÏ
 forward_rnn_25/gru_cell_25/add_2AddV2)forward_rnn_25/gru_cell_25/split:output:2"forward_rnn_25/gru_cell_25/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 forward_rnn_25/gru_cell_25/add_2£
forward_rnn_25/gru_cell_25/TanhTanh$forward_rnn_25/gru_cell_25/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
forward_rnn_25/gru_cell_25/TanhÅ
 forward_rnn_25/gru_cell_25/mul_1Mul&forward_rnn_25/gru_cell_25/Sigmoid:y:0forward_rnn_25/zeros:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 forward_rnn_25/gru_cell_25/mul_1
 forward_rnn_25/gru_cell_25/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2"
 forward_rnn_25/gru_cell_25/sub/xÍ
forward_rnn_25/gru_cell_25/subSub)forward_rnn_25/gru_cell_25/sub/x:output:0&forward_rnn_25/gru_cell_25/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
forward_rnn_25/gru_cell_25/subÇ
 forward_rnn_25/gru_cell_25/mul_2Mul"forward_rnn_25/gru_cell_25/sub:z:0#forward_rnn_25/gru_cell_25/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 forward_rnn_25/gru_cell_25/mul_2Ì
 forward_rnn_25/gru_cell_25/add_3AddV2$forward_rnn_25/gru_cell_25/mul_1:z:0$forward_rnn_25/gru_cell_25/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 forward_rnn_25/gru_cell_25/add_3­
,forward_rnn_25/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2.
,forward_rnn_25/TensorArrayV2_1/element_shapeô
forward_rnn_25/TensorArrayV2_1TensorListReserve5forward_rnn_25/TensorArrayV2_1/element_shape:output:0'forward_rnn_25/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
forward_rnn_25/TensorArrayV2_1l
forward_rnn_25/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
forward_rnn_25/time
'forward_rnn_25/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2)
'forward_rnn_25/while/maximum_iterations
!forward_rnn_25/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2#
!forward_rnn_25/while/loop_counter
forward_rnn_25/whileWhile*forward_rnn_25/while/loop_counter:output:00forward_rnn_25/while/maximum_iterations:output:0forward_rnn_25/time:output:0'forward_rnn_25/TensorArrayV2_1:handle:0forward_rnn_25/zeros:output:0'forward_rnn_25/strided_slice_1:output:0Fforward_rnn_25/TensorArrayUnstack/TensorListFromTensor:output_handle:02forward_rnn_25_gru_cell_25_readvariableop_resource9forward_rnn_25_gru_cell_25_matmul_readvariableop_resource;forward_rnn_25_gru_cell_25_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*/
body'R%
#forward_rnn_25_while_body_387381451*/
cond'R%
#forward_rnn_25_while_cond_387381450*9
output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
forward_rnn_25/whileÓ
?forward_rnn_25/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2A
?forward_rnn_25/TensorArrayV2Stack/TensorListStack/element_shape®
1forward_rnn_25/TensorArrayV2Stack/TensorListStackTensorListStackforward_rnn_25/while:output:3Hforward_rnn_25/TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype023
1forward_rnn_25/TensorArrayV2Stack/TensorListStack
$forward_rnn_25/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2&
$forward_rnn_25/strided_slice_3/stack
&forward_rnn_25/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&forward_rnn_25/strided_slice_3/stack_1
&forward_rnn_25/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&forward_rnn_25/strided_slice_3/stack_2õ
forward_rnn_25/strided_slice_3StridedSlice:forward_rnn_25/TensorArrayV2Stack/TensorListStack:tensor:0-forward_rnn_25/strided_slice_3/stack:output:0/forward_rnn_25/strided_slice_3/stack_1:output:0/forward_rnn_25/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2 
forward_rnn_25/strided_slice_3
forward_rnn_25/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
forward_rnn_25/transpose_1/permë
forward_rnn_25/transpose_1	Transpose:forward_rnn_25/TensorArrayV2Stack/TensorListStack:tensor:0(forward_rnn_25/transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
forward_rnn_25/transpose_1d
backward_rnn_25/ShapeShapeinputs*
T0*
_output_shapes
:2
backward_rnn_25/Shape
#backward_rnn_25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#backward_rnn_25/strided_slice/stack
%backward_rnn_25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%backward_rnn_25/strided_slice/stack_1
%backward_rnn_25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%backward_rnn_25/strided_slice/stack_2Â
backward_rnn_25/strided_sliceStridedSlicebackward_rnn_25/Shape:output:0,backward_rnn_25/strided_slice/stack:output:0.backward_rnn_25/strided_slice/stack_1:output:0.backward_rnn_25/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
backward_rnn_25/strided_slice}
backward_rnn_25/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
backward_rnn_25/zeros/mul/y¬
backward_rnn_25/zeros/mulMul&backward_rnn_25/strided_slice:output:0$backward_rnn_25/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
backward_rnn_25/zeros/mul
backward_rnn_25/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
backward_rnn_25/zeros/Less/y§
backward_rnn_25/zeros/LessLessbackward_rnn_25/zeros/mul:z:0%backward_rnn_25/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
backward_rnn_25/zeros/Less
backward_rnn_25/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2 
backward_rnn_25/zeros/packed/1Ã
backward_rnn_25/zeros/packedPack&backward_rnn_25/strided_slice:output:0'backward_rnn_25/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
backward_rnn_25/zeros/packed
backward_rnn_25/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
backward_rnn_25/zeros/Const¶
backward_rnn_25/zerosFill%backward_rnn_25/zeros/packed:output:0$backward_rnn_25/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
backward_rnn_25/zeros
backward_rnn_25/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
backward_rnn_25/transpose/perm´
backward_rnn_25/transpose	Transposeinputs'backward_rnn_25/transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
backward_rnn_25/transpose
backward_rnn_25/Shape_1Shapebackward_rnn_25/transpose:y:0*
T0*
_output_shapes
:2
backward_rnn_25/Shape_1
%backward_rnn_25/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_rnn_25/strided_slice_1/stack
'backward_rnn_25/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_rnn_25/strided_slice_1/stack_1
'backward_rnn_25/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_rnn_25/strided_slice_1/stack_2Î
backward_rnn_25/strided_slice_1StridedSlice backward_rnn_25/Shape_1:output:0.backward_rnn_25/strided_slice_1/stack:output:00backward_rnn_25/strided_slice_1/stack_1:output:00backward_rnn_25/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
backward_rnn_25/strided_slice_1¥
+backward_rnn_25/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2-
+backward_rnn_25/TensorArrayV2/element_shapeò
backward_rnn_25/TensorArrayV2TensorListReserve4backward_rnn_25/TensorArrayV2/element_shape:output:0(backward_rnn_25/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
backward_rnn_25/TensorArrayV2
backward_rnn_25/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2 
backward_rnn_25/ReverseV2/axisË
backward_rnn_25/ReverseV2	ReverseV2backward_rnn_25/transpose:y:0'backward_rnn_25/ReverseV2/axis:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
backward_rnn_25/ReverseV2ß
Ebackward_rnn_25/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2G
Ebackward_rnn_25/TensorArrayUnstack/TensorListFromTensor/element_shape½
7backward_rnn_25/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"backward_rnn_25/ReverseV2:output:0Nbackward_rnn_25/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7backward_rnn_25/TensorArrayUnstack/TensorListFromTensor
%backward_rnn_25/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%backward_rnn_25/strided_slice_2/stack
'backward_rnn_25/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_rnn_25/strided_slice_2/stack_1
'backward_rnn_25/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_rnn_25/strided_slice_2/stack_2Ý
backward_rnn_25/strided_slice_2StridedSlicebackward_rnn_25/transpose:y:0.backward_rnn_25/strided_slice_2/stack:output:00backward_rnn_25/strided_slice_2/stack_1:output:00backward_rnn_25/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2!
backward_rnn_25/strided_slice_2Í
*backward_rnn_25/gru_cell_25/ReadVariableOpReadVariableOp3backward_rnn_25_gru_cell_25_readvariableop_resource*
_output_shapes
:	*
dtype02,
*backward_rnn_25/gru_cell_25/ReadVariableOpÀ
#backward_rnn_25/gru_cell_25/unstackUnpack2backward_rnn_25/gru_cell_25/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2%
#backward_rnn_25/gru_cell_25/unstackã
1backward_rnn_25/gru_cell_25/MatMul/ReadVariableOpReadVariableOp:backward_rnn_25_gru_cell_25_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype023
1backward_rnn_25/gru_cell_25/MatMul/ReadVariableOpê
"backward_rnn_25/gru_cell_25/MatMulMatMul(backward_rnn_25/strided_slice_2:output:09backward_rnn_25/gru_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"backward_rnn_25/gru_cell_25/MatMulä
#backward_rnn_25/gru_cell_25/BiasAddBiasAdd,backward_rnn_25/gru_cell_25/MatMul:product:0,backward_rnn_25/gru_cell_25/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#backward_rnn_25/gru_cell_25/BiasAdd¥
+backward_rnn_25/gru_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2-
+backward_rnn_25/gru_cell_25/split/split_dim
!backward_rnn_25/gru_cell_25/splitSplit4backward_rnn_25/gru_cell_25/split/split_dim:output:0,backward_rnn_25/gru_cell_25/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2#
!backward_rnn_25/gru_cell_25/splité
3backward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOpReadVariableOp<backward_rnn_25_gru_cell_25_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype025
3backward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOpæ
$backward_rnn_25/gru_cell_25/MatMul_1MatMulbackward_rnn_25/zeros:output:0;backward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$backward_rnn_25/gru_cell_25/MatMul_1ê
%backward_rnn_25/gru_cell_25/BiasAdd_1BiasAdd.backward_rnn_25/gru_cell_25/MatMul_1:product:0,backward_rnn_25/gru_cell_25/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%backward_rnn_25/gru_cell_25/BiasAdd_1
!backward_rnn_25/gru_cell_25/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2#
!backward_rnn_25/gru_cell_25/Const©
-backward_rnn_25/gru_cell_25/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2/
-backward_rnn_25/gru_cell_25/split_1/split_dimà
#backward_rnn_25/gru_cell_25/split_1SplitV.backward_rnn_25/gru_cell_25/BiasAdd_1:output:0*backward_rnn_25/gru_cell_25/Const:output:06backward_rnn_25/gru_cell_25/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2%
#backward_rnn_25/gru_cell_25/split_1Ø
backward_rnn_25/gru_cell_25/addAddV2*backward_rnn_25/gru_cell_25/split:output:0,backward_rnn_25/gru_cell_25/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
backward_rnn_25/gru_cell_25/add­
#backward_rnn_25/gru_cell_25/SigmoidSigmoid#backward_rnn_25/gru_cell_25/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#backward_rnn_25/gru_cell_25/SigmoidÜ
!backward_rnn_25/gru_cell_25/add_1AddV2*backward_rnn_25/gru_cell_25/split:output:1,backward_rnn_25/gru_cell_25/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!backward_rnn_25/gru_cell_25/add_1³
%backward_rnn_25/gru_cell_25/Sigmoid_1Sigmoid%backward_rnn_25/gru_cell_25/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%backward_rnn_25/gru_cell_25/Sigmoid_1Õ
backward_rnn_25/gru_cell_25/mulMul)backward_rnn_25/gru_cell_25/Sigmoid_1:y:0,backward_rnn_25/gru_cell_25/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
backward_rnn_25/gru_cell_25/mulÓ
!backward_rnn_25/gru_cell_25/add_2AddV2*backward_rnn_25/gru_cell_25/split:output:2#backward_rnn_25/gru_cell_25/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!backward_rnn_25/gru_cell_25/add_2¦
 backward_rnn_25/gru_cell_25/TanhTanh%backward_rnn_25/gru_cell_25/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 backward_rnn_25/gru_cell_25/TanhÉ
!backward_rnn_25/gru_cell_25/mul_1Mul'backward_rnn_25/gru_cell_25/Sigmoid:y:0backward_rnn_25/zeros:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!backward_rnn_25/gru_cell_25/mul_1
!backward_rnn_25/gru_cell_25/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!backward_rnn_25/gru_cell_25/sub/xÑ
backward_rnn_25/gru_cell_25/subSub*backward_rnn_25/gru_cell_25/sub/x:output:0'backward_rnn_25/gru_cell_25/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
backward_rnn_25/gru_cell_25/subË
!backward_rnn_25/gru_cell_25/mul_2Mul#backward_rnn_25/gru_cell_25/sub:z:0$backward_rnn_25/gru_cell_25/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!backward_rnn_25/gru_cell_25/mul_2Ð
!backward_rnn_25/gru_cell_25/add_3AddV2%backward_rnn_25/gru_cell_25/mul_1:z:0%backward_rnn_25/gru_cell_25/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!backward_rnn_25/gru_cell_25/add_3¯
-backward_rnn_25/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2/
-backward_rnn_25/TensorArrayV2_1/element_shapeø
backward_rnn_25/TensorArrayV2_1TensorListReserve6backward_rnn_25/TensorArrayV2_1/element_shape:output:0(backward_rnn_25/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
backward_rnn_25/TensorArrayV2_1n
backward_rnn_25/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
backward_rnn_25/time
(backward_rnn_25/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2*
(backward_rnn_25/while/maximum_iterations
"backward_rnn_25/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2$
"backward_rnn_25/while/loop_counter
backward_rnn_25/whileWhile+backward_rnn_25/while/loop_counter:output:01backward_rnn_25/while/maximum_iterations:output:0backward_rnn_25/time:output:0(backward_rnn_25/TensorArrayV2_1:handle:0backward_rnn_25/zeros:output:0(backward_rnn_25/strided_slice_1:output:0Gbackward_rnn_25/TensorArrayUnstack/TensorListFromTensor:output_handle:03backward_rnn_25_gru_cell_25_readvariableop_resource:backward_rnn_25_gru_cell_25_matmul_readvariableop_resource<backward_rnn_25_gru_cell_25_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*0
body(R&
$backward_rnn_25_while_body_387381605*0
cond(R&
$backward_rnn_25_while_cond_387381604*9
output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
backward_rnn_25/whileÕ
@backward_rnn_25/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2B
@backward_rnn_25/TensorArrayV2Stack/TensorListStack/element_shape²
2backward_rnn_25/TensorArrayV2Stack/TensorListStackTensorListStackbackward_rnn_25/while:output:3Ibackward_rnn_25/TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype024
2backward_rnn_25/TensorArrayV2Stack/TensorListStack¡
%backward_rnn_25/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2'
%backward_rnn_25/strided_slice_3/stack
'backward_rnn_25/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'backward_rnn_25/strided_slice_3/stack_1
'backward_rnn_25/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'backward_rnn_25/strided_slice_3/stack_2û
backward_rnn_25/strided_slice_3StridedSlice;backward_rnn_25/TensorArrayV2Stack/TensorListStack:tensor:0.backward_rnn_25/strided_slice_3/stack:output:00backward_rnn_25/strided_slice_3/stack_1:output:00backward_rnn_25/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2!
backward_rnn_25/strided_slice_3
 backward_rnn_25/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 backward_rnn_25/transpose_1/permï
backward_rnn_25/transpose_1	Transpose;backward_rnn_25/TensorArrayV2Stack/TensorListStack:tensor:0)backward_rnn_25/transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
backward_rnn_25/transpose_1\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisÁ
concatConcatV2'forward_rnn_25/strided_slice_3:output:0(backward_rnn_25/strided_slice_3:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
concat¾
IdentityIdentityconcat:output:02^backward_rnn_25/gru_cell_25/MatMul/ReadVariableOp4^backward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp+^backward_rnn_25/gru_cell_25/ReadVariableOp^backward_rnn_25/while1^forward_rnn_25/gru_cell_25/MatMul/ReadVariableOp3^forward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp*^forward_rnn_25/gru_cell_25/ReadVariableOp^forward_rnn_25/while*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2f
1backward_rnn_25/gru_cell_25/MatMul/ReadVariableOp1backward_rnn_25/gru_cell_25/MatMul/ReadVariableOp2j
3backward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp3backward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp2X
*backward_rnn_25/gru_cell_25/ReadVariableOp*backward_rnn_25/gru_cell_25/ReadVariableOp2.
backward_rnn_25/whilebackward_rnn_25/while2d
0forward_rnn_25/gru_cell_25/MatMul/ReadVariableOp0forward_rnn_25/gru_cell_25/MatMul/ReadVariableOp2h
2forward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp2forward_rnn_25/gru_cell_25/MatMul_1/ReadVariableOp2V
)forward_rnn_25/gru_cell_25/ReadVariableOp)forward_rnn_25/gru_cell_25/ReadVariableOp2,
forward_rnn_25/whileforward_rnn_25/while:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
H

while_body_387382453
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_25_readvariableop_resource_0:	F
2while_gru_cell_25_matmul_readvariableop_resource_0:
H
4while_gru_cell_25_matmul_1_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_25_readvariableop_resource:	D
0while_gru_cell_25_matmul_readvariableop_resource:
F
2while_gru_cell_25_matmul_1_readvariableop_resource:
¢'while/gru_cell_25/MatMul/ReadVariableOp¢)while/gru_cell_25/MatMul_1/ReadVariableOp¢ while/gru_cell_25/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem±
 while/gru_cell_25/ReadVariableOpReadVariableOp+while_gru_cell_25_readvariableop_resource_0*
_output_shapes
:	*
dtype02"
 while/gru_cell_25/ReadVariableOp¢
while/gru_cell_25/unstackUnpack(while/gru_cell_25/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
while/gru_cell_25/unstackÇ
'while/gru_cell_25/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_25_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype02)
'while/gru_cell_25/MatMul/ReadVariableOpÔ
while/gru_cell_25/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/MatMul¼
while/gru_cell_25/BiasAddBiasAdd"while/gru_cell_25/MatMul:product:0"while/gru_cell_25/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/BiasAdd
!while/gru_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2#
!while/gru_cell_25/split/split_dim÷
while/gru_cell_25/splitSplit*while/gru_cell_25/split/split_dim:output:0"while/gru_cell_25/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/gru_cell_25/splitÍ
)while/gru_cell_25/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_25_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02+
)while/gru_cell_25/MatMul_1/ReadVariableOp½
while/gru_cell_25/MatMul_1MatMulwhile_placeholder_21while/gru_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/MatMul_1Â
while/gru_cell_25/BiasAdd_1BiasAdd$while/gru_cell_25/MatMul_1:product:0"while/gru_cell_25/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/BiasAdd_1
while/gru_cell_25/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2
while/gru_cell_25/Const
#while/gru_cell_25/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#while/gru_cell_25/split_1/split_dim®
while/gru_cell_25/split_1SplitV$while/gru_cell_25/BiasAdd_1:output:0 while/gru_cell_25/Const:output:0,while/gru_cell_25/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/gru_cell_25/split_1°
while/gru_cell_25/addAddV2 while/gru_cell_25/split:output:0"while/gru_cell_25/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/add
while/gru_cell_25/SigmoidSigmoidwhile/gru_cell_25/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/Sigmoid´
while/gru_cell_25/add_1AddV2 while/gru_cell_25/split:output:1"while/gru_cell_25/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/add_1
while/gru_cell_25/Sigmoid_1Sigmoidwhile/gru_cell_25/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/Sigmoid_1­
while/gru_cell_25/mulMulwhile/gru_cell_25/Sigmoid_1:y:0"while/gru_cell_25/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/mul«
while/gru_cell_25/add_2AddV2 while/gru_cell_25/split:output:2while/gru_cell_25/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/add_2
while/gru_cell_25/TanhTanhwhile/gru_cell_25/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/Tanh 
while/gru_cell_25/mul_1Mulwhile/gru_cell_25/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/mul_1w
while/gru_cell_25/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_25/sub/x©
while/gru_cell_25/subSub while/gru_cell_25/sub/x:output:0while/gru_cell_25/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/sub£
while/gru_cell_25/mul_2Mulwhile/gru_cell_25/sub:z:0while/gru_cell_25/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/mul_2¨
while/gru_cell_25/add_3AddV2while/gru_cell_25/mul_1:z:0while/gru_cell_25/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_25/add_3ß
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_25/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1×
while/IdentityIdentitywhile/add_1:z:0(^while/gru_cell_25/MatMul/ReadVariableOp*^while/gru_cell_25/MatMul_1/ReadVariableOp!^while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityê
while/Identity_1Identitywhile_while_maximum_iterations(^while/gru_cell_25/MatMul/ReadVariableOp*^while/gru_cell_25/MatMul_1/ReadVariableOp!^while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1Ù
while/Identity_2Identitywhile/add:z:0(^while/gru_cell_25/MatMul/ReadVariableOp*^while/gru_cell_25/MatMul_1/ReadVariableOp!^while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/gru_cell_25/MatMul/ReadVariableOp*^while/gru_cell_25/MatMul_1/ReadVariableOp!^while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3ù
while/Identity_4Identitywhile/gru_cell_25/add_3:z:0(^while/gru_cell_25/MatMul/ReadVariableOp*^while/gru_cell_25/MatMul_1/ReadVariableOp!^while/gru_cell_25/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4"j
2while_gru_cell_25_matmul_1_readvariableop_resource4while_gru_cell_25_matmul_1_readvariableop_resource_0"f
0while_gru_cell_25_matmul_readvariableop_resource2while_gru_cell_25_matmul_readvariableop_resource_0"X
)while_gru_cell_25_readvariableop_resource+while_gru_cell_25_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : 2R
'while/gru_cell_25/MatMul/ReadVariableOp'while/gru_cell_25/MatMul/ReadVariableOp2V
)while/gru_cell_25/MatMul_1/ReadVariableOp)while/gru_cell_25/MatMul_1/ReadVariableOp2D
 while/gru_cell_25/ReadVariableOp while/gru_cell_25/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 

¹
while_cond_387382116
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_17
3while_while_cond_387382116___redundant_placeholder07
3while_while_cond_387382116___redundant_placeholder17
3while_while_cond_387382116___redundant_placeholder27
3while_while_cond_387382116___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
î
Ý
K__inference_Tree_Encoder_layer_call_and_return_conditional_losses_387380079

inputs5
"double_bidirectional_gru_387380048:	6
"double_bidirectional_gru_387380050:
6
"double_bidirectional_gru_387380052:
5
"double_bidirectional_gru_387380054:	6
"double_bidirectional_gru_387380056:
6
"double_bidirectional_gru_387380058:
'
prediction_387380073:	"
prediction_387380075:
identity¢0Double_Bidirectional_GRU/StatefulPartitionedCall¢"Prediction/StatefulPartitionedCall
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
GPU2*0J 8 *Y
fTRR
P__inference_Embedding_Dropout_layer_call_and_return_conditional_losses_3873797332#
!Embedding_Dropout/PartitionedCall­
0Double_Bidirectional_GRU/StatefulPartitionedCallStatefulPartitionedCall*Embedding_Dropout/PartitionedCall:output:0"double_bidirectional_gru_387380048"double_bidirectional_gru_387380050"double_bidirectional_gru_387380052"double_bidirectional_gru_387380054"double_bidirectional_gru_387380056"double_bidirectional_gru_387380058*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *`
f[RY
W__inference_Double_Bidirectional_GRU_layer_call_and_return_conditional_losses_38738004722
0Double_Bidirectional_GRU/StatefulPartitionedCallÝ
"Prediction/StatefulPartitionedCallStatefulPartitionedCall9Double_Bidirectional_GRU/StatefulPartitionedCall:output:0prediction_387380073prediction_387380075*
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
GPU2*0J 8 *R
fMRK
I__inference_Prediction_layer_call_and_return_conditional_losses_3873800722$
"Prediction/StatefulPartitionedCall×
IdentityIdentity+Prediction/StatefulPartitionedCall:output:01^Double_Bidirectional_GRU/StatefulPartitionedCall#^Prediction/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : 2d
0Double_Bidirectional_GRU/StatefulPartitionedCall0Double_Bidirectional_GRU/StatefulPartitionedCall2H
"Prediction/StatefulPartitionedCall"Prediction/StatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ý
é
$backward_rnn_25_while_cond_387380349<
8backward_rnn_25_while_backward_rnn_25_while_loop_counterB
>backward_rnn_25_while_backward_rnn_25_while_maximum_iterations%
!backward_rnn_25_while_placeholder'
#backward_rnn_25_while_placeholder_1'
#backward_rnn_25_while_placeholder_2>
:backward_rnn_25_while_less_backward_rnn_25_strided_slice_1W
Sbackward_rnn_25_while_backward_rnn_25_while_cond_387380349___redundant_placeholder0W
Sbackward_rnn_25_while_backward_rnn_25_while_cond_387380349___redundant_placeholder1W
Sbackward_rnn_25_while_backward_rnn_25_while_cond_387380349___redundant_placeholder2W
Sbackward_rnn_25_while_backward_rnn_25_while_cond_387380349___redundant_placeholder3"
backward_rnn_25_while_identity
À
backward_rnn_25/while/LessLess!backward_rnn_25_while_placeholder:backward_rnn_25_while_less_backward_rnn_25_strided_slice_1*
T0*
_output_shapes
: 2
backward_rnn_25/while/Less
backward_rnn_25/while/IdentityIdentitybackward_rnn_25/while/Less:z:0*
T0
*
_output_shapes
: 2 
backward_rnn_25/while/Identity"I
backward_rnn_25_while_identity'backward_rnn_25/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
öZ

M__inference_forward_rnn_25_layer_call_and_return_conditional_losses_387382205
inputs_06
#gru_cell_25_readvariableop_resource:	>
*gru_cell_25_matmul_readvariableop_resource:
@
,gru_cell_25_matmul_1_readvariableop_resource:

identity¢!gru_cell_25/MatMul/ReadVariableOp¢#gru_cell_25/MatMul_1/ReadVariableOp¢gru_cell_25/ReadVariableOp¢whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2
gru_cell_25/ReadVariableOpReadVariableOp#gru_cell_25_readvariableop_resource*
_output_shapes
:	*
dtype02
gru_cell_25/ReadVariableOp
gru_cell_25/unstackUnpack"gru_cell_25/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
gru_cell_25/unstack³
!gru_cell_25/MatMul/ReadVariableOpReadVariableOp*gru_cell_25_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02#
!gru_cell_25/MatMul/ReadVariableOpª
gru_cell_25/MatMulMatMulstrided_slice_2:output:0)gru_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/MatMul¤
gru_cell_25/BiasAddBiasAddgru_cell_25/MatMul:product:0gru_cell_25/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/BiasAdd
gru_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/split/split_dimß
gru_cell_25/splitSplit$gru_cell_25/split/split_dim:output:0gru_cell_25/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
gru_cell_25/split¹
#gru_cell_25/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_25_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02%
#gru_cell_25/MatMul_1/ReadVariableOp¦
gru_cell_25/MatMul_1MatMulzeros:output:0+gru_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/MatMul_1ª
gru_cell_25/BiasAdd_1BiasAddgru_cell_25/MatMul_1:product:0gru_cell_25/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/BiasAdd_1{
gru_cell_25/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2
gru_cell_25/Const
gru_cell_25/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/split_1/split_dim
gru_cell_25/split_1SplitVgru_cell_25/BiasAdd_1:output:0gru_cell_25/Const:output:0&gru_cell_25/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
gru_cell_25/split_1
gru_cell_25/addAddV2gru_cell_25/split:output:0gru_cell_25/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/add}
gru_cell_25/SigmoidSigmoidgru_cell_25/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/Sigmoid
gru_cell_25/add_1AddV2gru_cell_25/split:output:1gru_cell_25/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/add_1
gru_cell_25/Sigmoid_1Sigmoidgru_cell_25/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/Sigmoid_1
gru_cell_25/mulMulgru_cell_25/Sigmoid_1:y:0gru_cell_25/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/mul
gru_cell_25/add_2AddV2gru_cell_25/split:output:2gru_cell_25/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/add_2v
gru_cell_25/TanhTanhgru_cell_25/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/Tanh
gru_cell_25/mul_1Mulgru_cell_25/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/mul_1k
gru_cell_25/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_25/sub/x
gru_cell_25/subSubgru_cell_25/sub/x:output:0gru_cell_25/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/sub
gru_cell_25/mul_2Mulgru_cell_25/sub:z:0gru_cell_25/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/mul_2
gru_cell_25/add_3AddV2gru_cell_25/mul_1:z:0gru_cell_25/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter²
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_25_readvariableop_resource*gru_cell_25_matmul_readvariableop_resource,gru_cell_25_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	* 
bodyR
while_body_387382117* 
condR
while_cond_387382116*9
output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
transpose_1Ü
IdentityIdentitystrided_slice_3:output:0"^gru_cell_25/MatMul/ReadVariableOp$^gru_cell_25/MatMul_1/ReadVariableOp^gru_cell_25/ReadVariableOp^while*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2F
!gru_cell_25/MatMul/ReadVariableOp!gru_cell_25/MatMul/ReadVariableOp2J
#gru_cell_25/MatMul_1/ReadVariableOp#gru_cell_25/MatMul_1/ReadVariableOp28
gru_cell_25/ReadVariableOpgru_cell_25/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
Äa
ï
$backward_rnn_25_while_body_387381917<
8backward_rnn_25_while_backward_rnn_25_while_loop_counterB
>backward_rnn_25_while_backward_rnn_25_while_maximum_iterations%
!backward_rnn_25_while_placeholder'
#backward_rnn_25_while_placeholder_1'
#backward_rnn_25_while_placeholder_2;
7backward_rnn_25_while_backward_rnn_25_strided_slice_1_0w
sbackward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_backward_rnn_25_tensorarrayunstack_tensorlistfromtensor_0N
;backward_rnn_25_while_gru_cell_25_readvariableop_resource_0:	V
Bbackward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource_0:
X
Dbackward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource_0:
"
backward_rnn_25_while_identity$
 backward_rnn_25_while_identity_1$
 backward_rnn_25_while_identity_2$
 backward_rnn_25_while_identity_3$
 backward_rnn_25_while_identity_49
5backward_rnn_25_while_backward_rnn_25_strided_slice_1u
qbackward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_backward_rnn_25_tensorarrayunstack_tensorlistfromtensorL
9backward_rnn_25_while_gru_cell_25_readvariableop_resource:	T
@backward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource:
V
Bbackward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource:
¢7backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp¢9backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp¢0backward_rnn_25/while/gru_cell_25/ReadVariableOpã
Gbackward_rnn_25/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2I
Gbackward_rnn_25/while/TensorArrayV2Read/TensorListGetItem/element_shape´
9backward_rnn_25/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsbackward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_backward_rnn_25_tensorarrayunstack_tensorlistfromtensor_0!backward_rnn_25_while_placeholderPbackward_rnn_25/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02;
9backward_rnn_25/while/TensorArrayV2Read/TensorListGetItemá
0backward_rnn_25/while/gru_cell_25/ReadVariableOpReadVariableOp;backward_rnn_25_while_gru_cell_25_readvariableop_resource_0*
_output_shapes
:	*
dtype022
0backward_rnn_25/while/gru_cell_25/ReadVariableOpÒ
)backward_rnn_25/while/gru_cell_25/unstackUnpack8backward_rnn_25/while/gru_cell_25/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2+
)backward_rnn_25/while/gru_cell_25/unstack÷
7backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOpReadVariableOpBbackward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype029
7backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp
(backward_rnn_25/while/gru_cell_25/MatMulMatMul@backward_rnn_25/while/TensorArrayV2Read/TensorListGetItem:item:0?backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(backward_rnn_25/while/gru_cell_25/MatMulü
)backward_rnn_25/while/gru_cell_25/BiasAddBiasAdd2backward_rnn_25/while/gru_cell_25/MatMul:product:02backward_rnn_25/while/gru_cell_25/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)backward_rnn_25/while/gru_cell_25/BiasAdd±
1backward_rnn_25/while/gru_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ23
1backward_rnn_25/while/gru_cell_25/split/split_dim·
'backward_rnn_25/while/gru_cell_25/splitSplit:backward_rnn_25/while/gru_cell_25/split/split_dim:output:02backward_rnn_25/while/gru_cell_25/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2)
'backward_rnn_25/while/gru_cell_25/splitý
9backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpReadVariableOpDbackward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02;
9backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpý
*backward_rnn_25/while/gru_cell_25/MatMul_1MatMul#backward_rnn_25_while_placeholder_2Abackward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*backward_rnn_25/while/gru_cell_25/MatMul_1
+backward_rnn_25/while/gru_cell_25/BiasAdd_1BiasAdd4backward_rnn_25/while/gru_cell_25/MatMul_1:product:02backward_rnn_25/while/gru_cell_25/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+backward_rnn_25/while/gru_cell_25/BiasAdd_1§
'backward_rnn_25/while/gru_cell_25/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2)
'backward_rnn_25/while/gru_cell_25/Constµ
3backward_rnn_25/while/gru_cell_25/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ25
3backward_rnn_25/while/gru_cell_25/split_1/split_dimþ
)backward_rnn_25/while/gru_cell_25/split_1SplitV4backward_rnn_25/while/gru_cell_25/BiasAdd_1:output:00backward_rnn_25/while/gru_cell_25/Const:output:0<backward_rnn_25/while/gru_cell_25/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2+
)backward_rnn_25/while/gru_cell_25/split_1ð
%backward_rnn_25/while/gru_cell_25/addAddV20backward_rnn_25/while/gru_cell_25/split:output:02backward_rnn_25/while/gru_cell_25/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%backward_rnn_25/while/gru_cell_25/add¿
)backward_rnn_25/while/gru_cell_25/SigmoidSigmoid)backward_rnn_25/while/gru_cell_25/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)backward_rnn_25/while/gru_cell_25/Sigmoidô
'backward_rnn_25/while/gru_cell_25/add_1AddV20backward_rnn_25/while/gru_cell_25/split:output:12backward_rnn_25/while/gru_cell_25/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'backward_rnn_25/while/gru_cell_25/add_1Å
+backward_rnn_25/while/gru_cell_25/Sigmoid_1Sigmoid+backward_rnn_25/while/gru_cell_25/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+backward_rnn_25/while/gru_cell_25/Sigmoid_1í
%backward_rnn_25/while/gru_cell_25/mulMul/backward_rnn_25/while/gru_cell_25/Sigmoid_1:y:02backward_rnn_25/while/gru_cell_25/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%backward_rnn_25/while/gru_cell_25/mulë
'backward_rnn_25/while/gru_cell_25/add_2AddV20backward_rnn_25/while/gru_cell_25/split:output:2)backward_rnn_25/while/gru_cell_25/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'backward_rnn_25/while/gru_cell_25/add_2¸
&backward_rnn_25/while/gru_cell_25/TanhTanh+backward_rnn_25/while/gru_cell_25/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&backward_rnn_25/while/gru_cell_25/Tanhà
'backward_rnn_25/while/gru_cell_25/mul_1Mul-backward_rnn_25/while/gru_cell_25/Sigmoid:y:0#backward_rnn_25_while_placeholder_2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'backward_rnn_25/while/gru_cell_25/mul_1
'backward_rnn_25/while/gru_cell_25/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'backward_rnn_25/while/gru_cell_25/sub/xé
%backward_rnn_25/while/gru_cell_25/subSub0backward_rnn_25/while/gru_cell_25/sub/x:output:0-backward_rnn_25/while/gru_cell_25/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%backward_rnn_25/while/gru_cell_25/subã
'backward_rnn_25/while/gru_cell_25/mul_2Mul)backward_rnn_25/while/gru_cell_25/sub:z:0*backward_rnn_25/while/gru_cell_25/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'backward_rnn_25/while/gru_cell_25/mul_2è
'backward_rnn_25/while/gru_cell_25/add_3AddV2+backward_rnn_25/while/gru_cell_25/mul_1:z:0+backward_rnn_25/while/gru_cell_25/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'backward_rnn_25/while/gru_cell_25/add_3¯
:backward_rnn_25/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#backward_rnn_25_while_placeholder_1!backward_rnn_25_while_placeholder+backward_rnn_25/while/gru_cell_25/add_3:z:0*
_output_shapes
: *
element_dtype02<
:backward_rnn_25/while/TensorArrayV2Write/TensorListSetItem|
backward_rnn_25/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_rnn_25/while/add/y©
backward_rnn_25/while/addAddV2!backward_rnn_25_while_placeholder$backward_rnn_25/while/add/y:output:0*
T0*
_output_shapes
: 2
backward_rnn_25/while/add
backward_rnn_25/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
backward_rnn_25/while/add_1/yÆ
backward_rnn_25/while/add_1AddV28backward_rnn_25_while_backward_rnn_25_while_loop_counter&backward_rnn_25/while/add_1/y:output:0*
T0*
_output_shapes
: 2
backward_rnn_25/while/add_1·
backward_rnn_25/while/IdentityIdentitybackward_rnn_25/while/add_1:z:08^backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp:^backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp1^backward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2 
backward_rnn_25/while/IdentityÚ
 backward_rnn_25/while/Identity_1Identity>backward_rnn_25_while_backward_rnn_25_while_maximum_iterations8^backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp:^backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp1^backward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2"
 backward_rnn_25/while/Identity_1¹
 backward_rnn_25/while/Identity_2Identitybackward_rnn_25/while/add:z:08^backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp:^backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp1^backward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2"
 backward_rnn_25/while/Identity_2æ
 backward_rnn_25/while/Identity_3IdentityJbackward_rnn_25/while/TensorArrayV2Write/TensorListSetItem:output_handle:08^backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp:^backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp1^backward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2"
 backward_rnn_25/while/Identity_3Ù
 backward_rnn_25/while/Identity_4Identity+backward_rnn_25/while/gru_cell_25/add_3:z:08^backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp:^backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp1^backward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 backward_rnn_25/while/Identity_4"p
5backward_rnn_25_while_backward_rnn_25_strided_slice_17backward_rnn_25_while_backward_rnn_25_strided_slice_1_0"
Bbackward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resourceDbackward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource_0"
@backward_rnn_25_while_gru_cell_25_matmul_readvariableop_resourceBbackward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource_0"x
9backward_rnn_25_while_gru_cell_25_readvariableop_resource;backward_rnn_25_while_gru_cell_25_readvariableop_resource_0"I
backward_rnn_25_while_identity'backward_rnn_25/while/Identity:output:0"M
 backward_rnn_25_while_identity_1)backward_rnn_25/while/Identity_1:output:0"M
 backward_rnn_25_while_identity_2)backward_rnn_25/while/Identity_2:output:0"M
 backward_rnn_25_while_identity_3)backward_rnn_25/while/Identity_3:output:0"M
 backward_rnn_25_while_identity_4)backward_rnn_25/while/Identity_4:output:0"è
qbackward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_backward_rnn_25_tensorarrayunstack_tensorlistfromtensorsbackward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_backward_rnn_25_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : 2r
7backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp7backward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp2v
9backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp9backward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp2d
0backward_rnn_25/while/gru_cell_25/ReadVariableOp0backward_rnn_25/while/gru_cell_25/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ö\

N__inference_backward_rnn_25_layer_call_and_return_conditional_losses_387382541
inputs_06
#gru_cell_25_readvariableop_resource:	>
*gru_cell_25_matmul_readvariableop_resource:
@
,gru_cell_25_matmul_1_readvariableop_resource:

identity¢!gru_cell_25/MatMul/ReadVariableOp¢#gru_cell_25/MatMul_1/ReadVariableOp¢gru_cell_25/ReadVariableOp¢whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2j
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2
ReverseV2/axis
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	ReverseV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeý
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2
gru_cell_25/ReadVariableOpReadVariableOp#gru_cell_25_readvariableop_resource*
_output_shapes
:	*
dtype02
gru_cell_25/ReadVariableOp
gru_cell_25/unstackUnpack"gru_cell_25/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
gru_cell_25/unstack³
!gru_cell_25/MatMul/ReadVariableOpReadVariableOp*gru_cell_25_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02#
!gru_cell_25/MatMul/ReadVariableOpª
gru_cell_25/MatMulMatMulstrided_slice_2:output:0)gru_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/MatMul¤
gru_cell_25/BiasAddBiasAddgru_cell_25/MatMul:product:0gru_cell_25/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/BiasAdd
gru_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/split/split_dimß
gru_cell_25/splitSplit$gru_cell_25/split/split_dim:output:0gru_cell_25/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
gru_cell_25/split¹
#gru_cell_25/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_25_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02%
#gru_cell_25/MatMul_1/ReadVariableOp¦
gru_cell_25/MatMul_1MatMulzeros:output:0+gru_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/MatMul_1ª
gru_cell_25/BiasAdd_1BiasAddgru_cell_25/MatMul_1:product:0gru_cell_25/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/BiasAdd_1{
gru_cell_25/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2
gru_cell_25/Const
gru_cell_25/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/split_1/split_dim
gru_cell_25/split_1SplitVgru_cell_25/BiasAdd_1:output:0gru_cell_25/Const:output:0&gru_cell_25/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
gru_cell_25/split_1
gru_cell_25/addAddV2gru_cell_25/split:output:0gru_cell_25/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/add}
gru_cell_25/SigmoidSigmoidgru_cell_25/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/Sigmoid
gru_cell_25/add_1AddV2gru_cell_25/split:output:1gru_cell_25/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/add_1
gru_cell_25/Sigmoid_1Sigmoidgru_cell_25/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/Sigmoid_1
gru_cell_25/mulMulgru_cell_25/Sigmoid_1:y:0gru_cell_25/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/mul
gru_cell_25/add_2AddV2gru_cell_25/split:output:2gru_cell_25/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/add_2v
gru_cell_25/TanhTanhgru_cell_25/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/Tanh
gru_cell_25/mul_1Mulgru_cell_25/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/mul_1k
gru_cell_25/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_25/sub/x
gru_cell_25/subSubgru_cell_25/sub/x:output:0gru_cell_25/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/sub
gru_cell_25/mul_2Mulgru_cell_25/sub:z:0gru_cell_25/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/mul_2
gru_cell_25/add_3AddV2gru_cell_25/mul_1:z:0gru_cell_25/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_25/add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter²
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_25_readvariableop_resource*gru_cell_25_matmul_readvariableop_resource,gru_cell_25_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	* 
bodyR
while_body_387382453* 
condR
while_cond_387382452*9
output_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
transpose_1Ü
IdentityIdentitystrided_slice_3:output:0"^gru_cell_25/MatMul/ReadVariableOp$^gru_cell_25/MatMul_1/ReadVariableOp^gru_cell_25/ReadVariableOp^while*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2F
!gru_cell_25/MatMul/ReadVariableOp!gru_cell_25/MatMul/ReadVariableOp2J
#gru_cell_25/MatMul_1/ReadVariableOp#gru_cell_25/MatMul_1/ReadVariableOp28
gru_cell_25/ReadVariableOpgru_cell_25/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
Ý
é
$backward_rnn_25_while_cond_387381604<
8backward_rnn_25_while_backward_rnn_25_while_loop_counterB
>backward_rnn_25_while_backward_rnn_25_while_maximum_iterations%
!backward_rnn_25_while_placeholder'
#backward_rnn_25_while_placeholder_1'
#backward_rnn_25_while_placeholder_2>
:backward_rnn_25_while_less_backward_rnn_25_strided_slice_1W
Sbackward_rnn_25_while_backward_rnn_25_while_cond_387381604___redundant_placeholder0W
Sbackward_rnn_25_while_backward_rnn_25_while_cond_387381604___redundant_placeholder1W
Sbackward_rnn_25_while_backward_rnn_25_while_cond_387381604___redundant_placeholder2W
Sbackward_rnn_25_while_backward_rnn_25_while_cond_387381604___redundant_placeholder3"
backward_rnn_25_while_identity
À
backward_rnn_25/while/LessLess!backward_rnn_25_while_placeholder:backward_rnn_25_while_less_backward_rnn_25_strided_slice_1*
T0*
_output_shapes
: 2
backward_rnn_25/while/Less
backward_rnn_25/while/IdentityIdentitybackward_rnn_25/while/Less:z:0*
T0
*
_output_shapes
: 2 
backward_rnn_25/while/Identity"I
backward_rnn_25_while_identity'backward_rnn_25/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Û	
Ö
0__inference_Tree_Encoder_layer_call_fn_387380098

inputs
unknown:	
	unknown_0:

	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:

	unknown_5:	
	unknown_6:
identity¢StatefulPartitionedCallÌ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_Tree_Encoder_layer_call_and_return_conditional_losses_3873800792
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameInputs
¾

K__inference_Tree_Encoder_layer_call_and_return_conditional_losses_387380610

inputs5
"double_bidirectional_gru_387380591:	6
"double_bidirectional_gru_387380593:
6
"double_bidirectional_gru_387380595:
5
"double_bidirectional_gru_387380597:	6
"double_bidirectional_gru_387380599:
6
"double_bidirectional_gru_387380601:
'
prediction_387380604:	"
prediction_387380606:
identity¢0Double_Bidirectional_GRU/StatefulPartitionedCall¢)Embedding_Dropout/StatefulPartitionedCall¢"Prediction/StatefulPartitionedCall
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
GPU2*0J 8 *Y
fTRR
P__inference_Embedding_Dropout_layer_call_and_return_conditional_losses_3873804752+
)Embedding_Dropout/StatefulPartitionedCallµ
0Double_Bidirectional_GRU/StatefulPartitionedCallStatefulPartitionedCall2Embedding_Dropout/StatefulPartitionedCall:output:0"double_bidirectional_gru_387380591"double_bidirectional_gru_387380593"double_bidirectional_gru_387380595"double_bidirectional_gru_387380597"double_bidirectional_gru_387380599"double_bidirectional_gru_387380601*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *`
f[RY
W__inference_Double_Bidirectional_GRU_layer_call_and_return_conditional_losses_38738044022
0Double_Bidirectional_GRU/StatefulPartitionedCallÝ
"Prediction/StatefulPartitionedCallStatefulPartitionedCall9Double_Bidirectional_GRU/StatefulPartitionedCall:output:0prediction_387380604prediction_387380606*
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
GPU2*0J 8 *R
fMRK
I__inference_Prediction_layer_call_and_return_conditional_losses_3873800722$
"Prediction/StatefulPartitionedCall
IdentityIdentity+Prediction/StatefulPartitionedCall:output:01^Double_Bidirectional_GRU/StatefulPartitionedCall*^Embedding_Dropout/StatefulPartitionedCall#^Prediction/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : 2d
0Double_Bidirectional_GRU/StatefulPartitionedCall0Double_Bidirectional_GRU/StatefulPartitionedCall2V
)Embedding_Dropout/StatefulPartitionedCall)Embedding_Dropout/StatefulPartitionedCall2H
"Prediction/StatefulPartitionedCall"Prediction/StatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameInputs

¹
while_cond_387382610
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_17
3while_while_cond_387382610___redundant_placeholder07
3while_while_cond_387382610___redundant_placeholder17
3while_while_cond_387382610___redundant_placeholder27
3while_while_cond_387382610___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Â

à
/__inference_gru_cell_25_layer_call_fn_387382727

inputs
states_0
unknown:	
	unknown_0:

	unknown_1:

identity

identity_1¢StatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_gru_cell_25_layer_call_and_return_conditional_losses_3873787732
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0
ø_
Ñ
#forward_rnn_25_while_body_387379803:
6forward_rnn_25_while_forward_rnn_25_while_loop_counter@
<forward_rnn_25_while_forward_rnn_25_while_maximum_iterations$
 forward_rnn_25_while_placeholder&
"forward_rnn_25_while_placeholder_1&
"forward_rnn_25_while_placeholder_29
5forward_rnn_25_while_forward_rnn_25_strided_slice_1_0u
qforward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_forward_rnn_25_tensorarrayunstack_tensorlistfromtensor_0M
:forward_rnn_25_while_gru_cell_25_readvariableop_resource_0:	U
Aforward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource_0:
W
Cforward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource_0:
!
forward_rnn_25_while_identity#
forward_rnn_25_while_identity_1#
forward_rnn_25_while_identity_2#
forward_rnn_25_while_identity_3#
forward_rnn_25_while_identity_47
3forward_rnn_25_while_forward_rnn_25_strided_slice_1s
oforward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_forward_rnn_25_tensorarrayunstack_tensorlistfromtensorK
8forward_rnn_25_while_gru_cell_25_readvariableop_resource:	S
?forward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource:
U
Aforward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource:
¢6forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp¢8forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp¢/forward_rnn_25/while/gru_cell_25/ReadVariableOpá
Fforward_rnn_25/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2H
Fforward_rnn_25/while/TensorArrayV2Read/TensorListGetItem/element_shape®
8forward_rnn_25/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemqforward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_forward_rnn_25_tensorarrayunstack_tensorlistfromtensor_0 forward_rnn_25_while_placeholderOforward_rnn_25/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02:
8forward_rnn_25/while/TensorArrayV2Read/TensorListGetItemÞ
/forward_rnn_25/while/gru_cell_25/ReadVariableOpReadVariableOp:forward_rnn_25_while_gru_cell_25_readvariableop_resource_0*
_output_shapes
:	*
dtype021
/forward_rnn_25/while/gru_cell_25/ReadVariableOpÏ
(forward_rnn_25/while/gru_cell_25/unstackUnpack7forward_rnn_25/while/gru_cell_25/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2*
(forward_rnn_25/while/gru_cell_25/unstackô
6forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOpReadVariableOpAforward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype028
6forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp
'forward_rnn_25/while/gru_cell_25/MatMulMatMul?forward_rnn_25/while/TensorArrayV2Read/TensorListGetItem:item:0>forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'forward_rnn_25/while/gru_cell_25/MatMulø
(forward_rnn_25/while/gru_cell_25/BiasAddBiasAdd1forward_rnn_25/while/gru_cell_25/MatMul:product:01forward_rnn_25/while/gru_cell_25/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(forward_rnn_25/while/gru_cell_25/BiasAdd¯
0forward_rnn_25/while/gru_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ22
0forward_rnn_25/while/gru_cell_25/split/split_dim³
&forward_rnn_25/while/gru_cell_25/splitSplit9forward_rnn_25/while/gru_cell_25/split/split_dim:output:01forward_rnn_25/while/gru_cell_25/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2(
&forward_rnn_25/while/gru_cell_25/splitú
8forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpReadVariableOpCforward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02:
8forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpù
)forward_rnn_25/while/gru_cell_25/MatMul_1MatMul"forward_rnn_25_while_placeholder_2@forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)forward_rnn_25/while/gru_cell_25/MatMul_1þ
*forward_rnn_25/while/gru_cell_25/BiasAdd_1BiasAdd3forward_rnn_25/while/gru_cell_25/MatMul_1:product:01forward_rnn_25/while/gru_cell_25/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*forward_rnn_25/while/gru_cell_25/BiasAdd_1¥
&forward_rnn_25/while/gru_cell_25/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2(
&forward_rnn_25/while/gru_cell_25/Const³
2forward_rnn_25/while/gru_cell_25/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ24
2forward_rnn_25/while/gru_cell_25/split_1/split_dimù
(forward_rnn_25/while/gru_cell_25/split_1SplitV3forward_rnn_25/while/gru_cell_25/BiasAdd_1:output:0/forward_rnn_25/while/gru_cell_25/Const:output:0;forward_rnn_25/while/gru_cell_25/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2*
(forward_rnn_25/while/gru_cell_25/split_1ì
$forward_rnn_25/while/gru_cell_25/addAddV2/forward_rnn_25/while/gru_cell_25/split:output:01forward_rnn_25/while/gru_cell_25/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$forward_rnn_25/while/gru_cell_25/add¼
(forward_rnn_25/while/gru_cell_25/SigmoidSigmoid(forward_rnn_25/while/gru_cell_25/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(forward_rnn_25/while/gru_cell_25/Sigmoidð
&forward_rnn_25/while/gru_cell_25/add_1AddV2/forward_rnn_25/while/gru_cell_25/split:output:11forward_rnn_25/while/gru_cell_25/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&forward_rnn_25/while/gru_cell_25/add_1Â
*forward_rnn_25/while/gru_cell_25/Sigmoid_1Sigmoid*forward_rnn_25/while/gru_cell_25/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*forward_rnn_25/while/gru_cell_25/Sigmoid_1é
$forward_rnn_25/while/gru_cell_25/mulMul.forward_rnn_25/while/gru_cell_25/Sigmoid_1:y:01forward_rnn_25/while/gru_cell_25/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$forward_rnn_25/while/gru_cell_25/mulç
&forward_rnn_25/while/gru_cell_25/add_2AddV2/forward_rnn_25/while/gru_cell_25/split:output:2(forward_rnn_25/while/gru_cell_25/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&forward_rnn_25/while/gru_cell_25/add_2µ
%forward_rnn_25/while/gru_cell_25/TanhTanh*forward_rnn_25/while/gru_cell_25/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%forward_rnn_25/while/gru_cell_25/TanhÜ
&forward_rnn_25/while/gru_cell_25/mul_1Mul,forward_rnn_25/while/gru_cell_25/Sigmoid:y:0"forward_rnn_25_while_placeholder_2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&forward_rnn_25/while/gru_cell_25/mul_1
&forward_rnn_25/while/gru_cell_25/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2(
&forward_rnn_25/while/gru_cell_25/sub/xå
$forward_rnn_25/while/gru_cell_25/subSub/forward_rnn_25/while/gru_cell_25/sub/x:output:0,forward_rnn_25/while/gru_cell_25/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$forward_rnn_25/while/gru_cell_25/subß
&forward_rnn_25/while/gru_cell_25/mul_2Mul(forward_rnn_25/while/gru_cell_25/sub:z:0)forward_rnn_25/while/gru_cell_25/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&forward_rnn_25/while/gru_cell_25/mul_2ä
&forward_rnn_25/while/gru_cell_25/add_3AddV2*forward_rnn_25/while/gru_cell_25/mul_1:z:0*forward_rnn_25/while/gru_cell_25/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&forward_rnn_25/while/gru_cell_25/add_3ª
9forward_rnn_25/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem"forward_rnn_25_while_placeholder_1 forward_rnn_25_while_placeholder*forward_rnn_25/while/gru_cell_25/add_3:z:0*
_output_shapes
: *
element_dtype02;
9forward_rnn_25/while/TensorArrayV2Write/TensorListSetItemz
forward_rnn_25/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_rnn_25/while/add/y¥
forward_rnn_25/while/addAddV2 forward_rnn_25_while_placeholder#forward_rnn_25/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_rnn_25/while/add~
forward_rnn_25/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_rnn_25/while/add_1/yÁ
forward_rnn_25/while/add_1AddV26forward_rnn_25_while_forward_rnn_25_while_loop_counter%forward_rnn_25/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_rnn_25/while/add_1±
forward_rnn_25/while/IdentityIdentityforward_rnn_25/while/add_1:z:07^forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp9^forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp0^forward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2
forward_rnn_25/while/IdentityÓ
forward_rnn_25/while/Identity_1Identity<forward_rnn_25_while_forward_rnn_25_while_maximum_iterations7^forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp9^forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp0^forward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2!
forward_rnn_25/while/Identity_1³
forward_rnn_25/while/Identity_2Identityforward_rnn_25/while/add:z:07^forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp9^forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp0^forward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2!
forward_rnn_25/while/Identity_2à
forward_rnn_25/while/Identity_3IdentityIforward_rnn_25/while/TensorArrayV2Write/TensorListSetItem:output_handle:07^forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp9^forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp0^forward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2!
forward_rnn_25/while/Identity_3Ó
forward_rnn_25/while/Identity_4Identity*forward_rnn_25/while/gru_cell_25/add_3:z:07^forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp9^forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp0^forward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
forward_rnn_25/while/Identity_4"l
3forward_rnn_25_while_forward_rnn_25_strided_slice_15forward_rnn_25_while_forward_rnn_25_strided_slice_1_0"
Aforward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resourceCforward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource_0"
?forward_rnn_25_while_gru_cell_25_matmul_readvariableop_resourceAforward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource_0"v
8forward_rnn_25_while_gru_cell_25_readvariableop_resource:forward_rnn_25_while_gru_cell_25_readvariableop_resource_0"G
forward_rnn_25_while_identity&forward_rnn_25/while/Identity:output:0"K
forward_rnn_25_while_identity_1(forward_rnn_25/while/Identity_1:output:0"K
forward_rnn_25_while_identity_2(forward_rnn_25/while/Identity_2:output:0"K
forward_rnn_25_while_identity_3(forward_rnn_25/while/Identity_3:output:0"K
forward_rnn_25_while_identity_4(forward_rnn_25/while/Identity_4:output:0"ä
oforward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_forward_rnn_25_tensorarrayunstack_tensorlistfromtensorqforward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_forward_rnn_25_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : 2p
6forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp6forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp2t
8forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp8forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp2b
/forward_rnn_25/while/gru_cell_25/ReadVariableOp/forward_rnn_25/while/gru_cell_25/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Û	
Ö
0__inference_Tree_Encoder_layer_call_fn_387380654

inputs
unknown:	
	unknown_0:

	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:

	unknown_5:	
	unknown_6:
identity¢StatefulPartitionedCallÌ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_Tree_Encoder_layer_call_and_return_conditional_losses_3873800792
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬#
Î
while_body_387378840
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_gru_cell_25_387378862_0:	1
while_gru_cell_25_387378864_0:
1
while_gru_cell_25_387378866_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_gru_cell_25_387378862:	/
while_gru_cell_25_387378864:
/
while_gru_cell_25_387378866:
¢)while/gru_cell_25/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÂ
)while/gru_cell_25/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_25_387378862_0while_gru_cell_25_387378864_0while_gru_cell_25_387378866_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_gru_cell_25_layer_call_and_return_conditional_losses_3873787732+
)while/gru_cell_25/StatefulPartitionedCallö
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_25/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:0*^while/gru_cell_25/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations*^while/gru_cell_25/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0*^while/gru_cell_25/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¹
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/gru_cell_25/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3Ã
while/Identity_4Identity2while/gru_cell_25/StatefulPartitionedCall:output:1*^while/gru_cell_25/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4"<
while_gru_cell_25_387378862while_gru_cell_25_387378862_0"<
while_gru_cell_25_387378864while_gru_cell_25_387378864_0"<
while_gru_cell_25_387378866while_gru_cell_25_387378866_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/gru_cell_25/StatefulPartitionedCall)while/gru_cell_25/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
©
o
P__inference_Embedding_Dropout_layer_call_and_return_conditional_losses_387381349

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
×
Þ
J__inference_gru_cell_25_layer_call_and_return_conditional_losses_387379353

inputs

states*
readvariableop_resource:	2
matmul_readvariableop_resource:
4
 matmul_1_readvariableop_resource:

identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpy
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
split/split_dim¯
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	BiasAdd_1c
ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2
Constq
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
split_1/split_dimÔ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanh]
mul_1MulSigmoid:y:0states*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sub[
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_3¡
IdentityIdentity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¥

Identity_1Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
«	
Í
'__inference_signature_wrapper_387380633

inputs
unknown:	
	unknown_0:

	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:

	unknown_5:	
	unknown_6:
identity¢StatefulPartitionedCall¥
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference__wrapped_model_3873785572
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameInputs
ø_
Ñ
#forward_rnn_25_while_body_387380196:
6forward_rnn_25_while_forward_rnn_25_while_loop_counter@
<forward_rnn_25_while_forward_rnn_25_while_maximum_iterations$
 forward_rnn_25_while_placeholder&
"forward_rnn_25_while_placeholder_1&
"forward_rnn_25_while_placeholder_29
5forward_rnn_25_while_forward_rnn_25_strided_slice_1_0u
qforward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_forward_rnn_25_tensorarrayunstack_tensorlistfromtensor_0M
:forward_rnn_25_while_gru_cell_25_readvariableop_resource_0:	U
Aforward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource_0:
W
Cforward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource_0:
!
forward_rnn_25_while_identity#
forward_rnn_25_while_identity_1#
forward_rnn_25_while_identity_2#
forward_rnn_25_while_identity_3#
forward_rnn_25_while_identity_47
3forward_rnn_25_while_forward_rnn_25_strided_slice_1s
oforward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_forward_rnn_25_tensorarrayunstack_tensorlistfromtensorK
8forward_rnn_25_while_gru_cell_25_readvariableop_resource:	S
?forward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource:
U
Aforward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource:
¢6forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp¢8forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp¢/forward_rnn_25/while/gru_cell_25/ReadVariableOpá
Fforward_rnn_25/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2H
Fforward_rnn_25/while/TensorArrayV2Read/TensorListGetItem/element_shape®
8forward_rnn_25/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemqforward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_forward_rnn_25_tensorarrayunstack_tensorlistfromtensor_0 forward_rnn_25_while_placeholderOforward_rnn_25/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02:
8forward_rnn_25/while/TensorArrayV2Read/TensorListGetItemÞ
/forward_rnn_25/while/gru_cell_25/ReadVariableOpReadVariableOp:forward_rnn_25_while_gru_cell_25_readvariableop_resource_0*
_output_shapes
:	*
dtype021
/forward_rnn_25/while/gru_cell_25/ReadVariableOpÏ
(forward_rnn_25/while/gru_cell_25/unstackUnpack7forward_rnn_25/while/gru_cell_25/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2*
(forward_rnn_25/while/gru_cell_25/unstackô
6forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOpReadVariableOpAforward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype028
6forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp
'forward_rnn_25/while/gru_cell_25/MatMulMatMul?forward_rnn_25/while/TensorArrayV2Read/TensorListGetItem:item:0>forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'forward_rnn_25/while/gru_cell_25/MatMulø
(forward_rnn_25/while/gru_cell_25/BiasAddBiasAdd1forward_rnn_25/while/gru_cell_25/MatMul:product:01forward_rnn_25/while/gru_cell_25/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(forward_rnn_25/while/gru_cell_25/BiasAdd¯
0forward_rnn_25/while/gru_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ22
0forward_rnn_25/while/gru_cell_25/split/split_dim³
&forward_rnn_25/while/gru_cell_25/splitSplit9forward_rnn_25/while/gru_cell_25/split/split_dim:output:01forward_rnn_25/while/gru_cell_25/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2(
&forward_rnn_25/while/gru_cell_25/splitú
8forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpReadVariableOpCforward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02:
8forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpù
)forward_rnn_25/while/gru_cell_25/MatMul_1MatMul"forward_rnn_25_while_placeholder_2@forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)forward_rnn_25/while/gru_cell_25/MatMul_1þ
*forward_rnn_25/while/gru_cell_25/BiasAdd_1BiasAdd3forward_rnn_25/while/gru_cell_25/MatMul_1:product:01forward_rnn_25/while/gru_cell_25/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*forward_rnn_25/while/gru_cell_25/BiasAdd_1¥
&forward_rnn_25/while/gru_cell_25/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2(
&forward_rnn_25/while/gru_cell_25/Const³
2forward_rnn_25/while/gru_cell_25/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ24
2forward_rnn_25/while/gru_cell_25/split_1/split_dimù
(forward_rnn_25/while/gru_cell_25/split_1SplitV3forward_rnn_25/while/gru_cell_25/BiasAdd_1:output:0/forward_rnn_25/while/gru_cell_25/Const:output:0;forward_rnn_25/while/gru_cell_25/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2*
(forward_rnn_25/while/gru_cell_25/split_1ì
$forward_rnn_25/while/gru_cell_25/addAddV2/forward_rnn_25/while/gru_cell_25/split:output:01forward_rnn_25/while/gru_cell_25/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$forward_rnn_25/while/gru_cell_25/add¼
(forward_rnn_25/while/gru_cell_25/SigmoidSigmoid(forward_rnn_25/while/gru_cell_25/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(forward_rnn_25/while/gru_cell_25/Sigmoidð
&forward_rnn_25/while/gru_cell_25/add_1AddV2/forward_rnn_25/while/gru_cell_25/split:output:11forward_rnn_25/while/gru_cell_25/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&forward_rnn_25/while/gru_cell_25/add_1Â
*forward_rnn_25/while/gru_cell_25/Sigmoid_1Sigmoid*forward_rnn_25/while/gru_cell_25/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*forward_rnn_25/while/gru_cell_25/Sigmoid_1é
$forward_rnn_25/while/gru_cell_25/mulMul.forward_rnn_25/while/gru_cell_25/Sigmoid_1:y:01forward_rnn_25/while/gru_cell_25/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$forward_rnn_25/while/gru_cell_25/mulç
&forward_rnn_25/while/gru_cell_25/add_2AddV2/forward_rnn_25/while/gru_cell_25/split:output:2(forward_rnn_25/while/gru_cell_25/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&forward_rnn_25/while/gru_cell_25/add_2µ
%forward_rnn_25/while/gru_cell_25/TanhTanh*forward_rnn_25/while/gru_cell_25/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%forward_rnn_25/while/gru_cell_25/TanhÜ
&forward_rnn_25/while/gru_cell_25/mul_1Mul,forward_rnn_25/while/gru_cell_25/Sigmoid:y:0"forward_rnn_25_while_placeholder_2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&forward_rnn_25/while/gru_cell_25/mul_1
&forward_rnn_25/while/gru_cell_25/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2(
&forward_rnn_25/while/gru_cell_25/sub/xå
$forward_rnn_25/while/gru_cell_25/subSub/forward_rnn_25/while/gru_cell_25/sub/x:output:0,forward_rnn_25/while/gru_cell_25/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$forward_rnn_25/while/gru_cell_25/subß
&forward_rnn_25/while/gru_cell_25/mul_2Mul(forward_rnn_25/while/gru_cell_25/sub:z:0)forward_rnn_25/while/gru_cell_25/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&forward_rnn_25/while/gru_cell_25/mul_2ä
&forward_rnn_25/while/gru_cell_25/add_3AddV2*forward_rnn_25/while/gru_cell_25/mul_1:z:0*forward_rnn_25/while/gru_cell_25/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&forward_rnn_25/while/gru_cell_25/add_3ª
9forward_rnn_25/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem"forward_rnn_25_while_placeholder_1 forward_rnn_25_while_placeholder*forward_rnn_25/while/gru_cell_25/add_3:z:0*
_output_shapes
: *
element_dtype02;
9forward_rnn_25/while/TensorArrayV2Write/TensorListSetItemz
forward_rnn_25/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_rnn_25/while/add/y¥
forward_rnn_25/while/addAddV2 forward_rnn_25_while_placeholder#forward_rnn_25/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_rnn_25/while/add~
forward_rnn_25/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_rnn_25/while/add_1/yÁ
forward_rnn_25/while/add_1AddV26forward_rnn_25_while_forward_rnn_25_while_loop_counter%forward_rnn_25/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_rnn_25/while/add_1±
forward_rnn_25/while/IdentityIdentityforward_rnn_25/while/add_1:z:07^forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp9^forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp0^forward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2
forward_rnn_25/while/IdentityÓ
forward_rnn_25/while/Identity_1Identity<forward_rnn_25_while_forward_rnn_25_while_maximum_iterations7^forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp9^forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp0^forward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2!
forward_rnn_25/while/Identity_1³
forward_rnn_25/while/Identity_2Identityforward_rnn_25/while/add:z:07^forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp9^forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp0^forward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2!
forward_rnn_25/while/Identity_2à
forward_rnn_25/while/Identity_3IdentityIforward_rnn_25/while/TensorArrayV2Write/TensorListSetItem:output_handle:07^forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp9^forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp0^forward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2!
forward_rnn_25/while/Identity_3Ó
forward_rnn_25/while/Identity_4Identity*forward_rnn_25/while/gru_cell_25/add_3:z:07^forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp9^forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp0^forward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
forward_rnn_25/while/Identity_4"l
3forward_rnn_25_while_forward_rnn_25_strided_slice_15forward_rnn_25_while_forward_rnn_25_strided_slice_1_0"
Aforward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resourceCforward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource_0"
?forward_rnn_25_while_gru_cell_25_matmul_readvariableop_resourceAforward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource_0"v
8forward_rnn_25_while_gru_cell_25_readvariableop_resource:forward_rnn_25_while_gru_cell_25_readvariableop_resource_0"G
forward_rnn_25_while_identity&forward_rnn_25/while/Identity:output:0"K
forward_rnn_25_while_identity_1(forward_rnn_25/while/Identity_1:output:0"K
forward_rnn_25_while_identity_2(forward_rnn_25/while/Identity_2:output:0"K
forward_rnn_25_while_identity_3(forward_rnn_25/while/Identity_3:output:0"K
forward_rnn_25_while_identity_4(forward_rnn_25/while/Identity_4:output:0"ä
oforward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_forward_rnn_25_tensorarrayunstack_tensorlistfromtensorqforward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_forward_rnn_25_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : 2p
6forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp6forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp2t
8forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp8forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp2b
/forward_rnn_25/while/gru_cell_25/ReadVariableOp/forward_rnn_25/while/gru_cell_25/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 

¹
while_cond_387382452
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_17
3while_while_cond_387382452___redundant_placeholder07
3while_while_cond_387382452___redundant_placeholder17
3while_while_cond_387382452___redundant_placeholder27
3while_while_cond_387382452___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ø_
Ñ
#forward_rnn_25_while_body_387381451:
6forward_rnn_25_while_forward_rnn_25_while_loop_counter@
<forward_rnn_25_while_forward_rnn_25_while_maximum_iterations$
 forward_rnn_25_while_placeholder&
"forward_rnn_25_while_placeholder_1&
"forward_rnn_25_while_placeholder_29
5forward_rnn_25_while_forward_rnn_25_strided_slice_1_0u
qforward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_forward_rnn_25_tensorarrayunstack_tensorlistfromtensor_0M
:forward_rnn_25_while_gru_cell_25_readvariableop_resource_0:	U
Aforward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource_0:
W
Cforward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource_0:
!
forward_rnn_25_while_identity#
forward_rnn_25_while_identity_1#
forward_rnn_25_while_identity_2#
forward_rnn_25_while_identity_3#
forward_rnn_25_while_identity_47
3forward_rnn_25_while_forward_rnn_25_strided_slice_1s
oforward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_forward_rnn_25_tensorarrayunstack_tensorlistfromtensorK
8forward_rnn_25_while_gru_cell_25_readvariableop_resource:	S
?forward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource:
U
Aforward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource:
¢6forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp¢8forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp¢/forward_rnn_25/while/gru_cell_25/ReadVariableOpá
Fforward_rnn_25/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2H
Fforward_rnn_25/while/TensorArrayV2Read/TensorListGetItem/element_shape®
8forward_rnn_25/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemqforward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_forward_rnn_25_tensorarrayunstack_tensorlistfromtensor_0 forward_rnn_25_while_placeholderOforward_rnn_25/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02:
8forward_rnn_25/while/TensorArrayV2Read/TensorListGetItemÞ
/forward_rnn_25/while/gru_cell_25/ReadVariableOpReadVariableOp:forward_rnn_25_while_gru_cell_25_readvariableop_resource_0*
_output_shapes
:	*
dtype021
/forward_rnn_25/while/gru_cell_25/ReadVariableOpÏ
(forward_rnn_25/while/gru_cell_25/unstackUnpack7forward_rnn_25/while/gru_cell_25/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2*
(forward_rnn_25/while/gru_cell_25/unstackô
6forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOpReadVariableOpAforward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype028
6forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp
'forward_rnn_25/while/gru_cell_25/MatMulMatMul?forward_rnn_25/while/TensorArrayV2Read/TensorListGetItem:item:0>forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'forward_rnn_25/while/gru_cell_25/MatMulø
(forward_rnn_25/while/gru_cell_25/BiasAddBiasAdd1forward_rnn_25/while/gru_cell_25/MatMul:product:01forward_rnn_25/while/gru_cell_25/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(forward_rnn_25/while/gru_cell_25/BiasAdd¯
0forward_rnn_25/while/gru_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ22
0forward_rnn_25/while/gru_cell_25/split/split_dim³
&forward_rnn_25/while/gru_cell_25/splitSplit9forward_rnn_25/while/gru_cell_25/split/split_dim:output:01forward_rnn_25/while/gru_cell_25/BiasAdd:output:0*
T0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2(
&forward_rnn_25/while/gru_cell_25/splitú
8forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpReadVariableOpCforward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02:
8forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOpù
)forward_rnn_25/while/gru_cell_25/MatMul_1MatMul"forward_rnn_25_while_placeholder_2@forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)forward_rnn_25/while/gru_cell_25/MatMul_1þ
*forward_rnn_25/while/gru_cell_25/BiasAdd_1BiasAdd3forward_rnn_25/while/gru_cell_25/MatMul_1:product:01forward_rnn_25/while/gru_cell_25/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*forward_rnn_25/while/gru_cell_25/BiasAdd_1¥
&forward_rnn_25/while/gru_cell_25/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2(
&forward_rnn_25/while/gru_cell_25/Const³
2forward_rnn_25/while/gru_cell_25/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ24
2forward_rnn_25/while/gru_cell_25/split_1/split_dimù
(forward_rnn_25/while/gru_cell_25/split_1SplitV3forward_rnn_25/while/gru_cell_25/BiasAdd_1:output:0/forward_rnn_25/while/gru_cell_25/Const:output:0;forward_rnn_25/while/gru_cell_25/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2*
(forward_rnn_25/while/gru_cell_25/split_1ì
$forward_rnn_25/while/gru_cell_25/addAddV2/forward_rnn_25/while/gru_cell_25/split:output:01forward_rnn_25/while/gru_cell_25/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$forward_rnn_25/while/gru_cell_25/add¼
(forward_rnn_25/while/gru_cell_25/SigmoidSigmoid(forward_rnn_25/while/gru_cell_25/add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(forward_rnn_25/while/gru_cell_25/Sigmoidð
&forward_rnn_25/while/gru_cell_25/add_1AddV2/forward_rnn_25/while/gru_cell_25/split:output:11forward_rnn_25/while/gru_cell_25/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&forward_rnn_25/while/gru_cell_25/add_1Â
*forward_rnn_25/while/gru_cell_25/Sigmoid_1Sigmoid*forward_rnn_25/while/gru_cell_25/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*forward_rnn_25/while/gru_cell_25/Sigmoid_1é
$forward_rnn_25/while/gru_cell_25/mulMul.forward_rnn_25/while/gru_cell_25/Sigmoid_1:y:01forward_rnn_25/while/gru_cell_25/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$forward_rnn_25/while/gru_cell_25/mulç
&forward_rnn_25/while/gru_cell_25/add_2AddV2/forward_rnn_25/while/gru_cell_25/split:output:2(forward_rnn_25/while/gru_cell_25/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&forward_rnn_25/while/gru_cell_25/add_2µ
%forward_rnn_25/while/gru_cell_25/TanhTanh*forward_rnn_25/while/gru_cell_25/add_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%forward_rnn_25/while/gru_cell_25/TanhÜ
&forward_rnn_25/while/gru_cell_25/mul_1Mul,forward_rnn_25/while/gru_cell_25/Sigmoid:y:0"forward_rnn_25_while_placeholder_2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&forward_rnn_25/while/gru_cell_25/mul_1
&forward_rnn_25/while/gru_cell_25/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2(
&forward_rnn_25/while/gru_cell_25/sub/xå
$forward_rnn_25/while/gru_cell_25/subSub/forward_rnn_25/while/gru_cell_25/sub/x:output:0,forward_rnn_25/while/gru_cell_25/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$forward_rnn_25/while/gru_cell_25/subß
&forward_rnn_25/while/gru_cell_25/mul_2Mul(forward_rnn_25/while/gru_cell_25/sub:z:0)forward_rnn_25/while/gru_cell_25/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&forward_rnn_25/while/gru_cell_25/mul_2ä
&forward_rnn_25/while/gru_cell_25/add_3AddV2*forward_rnn_25/while/gru_cell_25/mul_1:z:0*forward_rnn_25/while/gru_cell_25/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&forward_rnn_25/while/gru_cell_25/add_3ª
9forward_rnn_25/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem"forward_rnn_25_while_placeholder_1 forward_rnn_25_while_placeholder*forward_rnn_25/while/gru_cell_25/add_3:z:0*
_output_shapes
: *
element_dtype02;
9forward_rnn_25/while/TensorArrayV2Write/TensorListSetItemz
forward_rnn_25/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_rnn_25/while/add/y¥
forward_rnn_25/while/addAddV2 forward_rnn_25_while_placeholder#forward_rnn_25/while/add/y:output:0*
T0*
_output_shapes
: 2
forward_rnn_25/while/add~
forward_rnn_25/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
forward_rnn_25/while/add_1/yÁ
forward_rnn_25/while/add_1AddV26forward_rnn_25_while_forward_rnn_25_while_loop_counter%forward_rnn_25/while/add_1/y:output:0*
T0*
_output_shapes
: 2
forward_rnn_25/while/add_1±
forward_rnn_25/while/IdentityIdentityforward_rnn_25/while/add_1:z:07^forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp9^forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp0^forward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2
forward_rnn_25/while/IdentityÓ
forward_rnn_25/while/Identity_1Identity<forward_rnn_25_while_forward_rnn_25_while_maximum_iterations7^forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp9^forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp0^forward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2!
forward_rnn_25/while/Identity_1³
forward_rnn_25/while/Identity_2Identityforward_rnn_25/while/add:z:07^forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp9^forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp0^forward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2!
forward_rnn_25/while/Identity_2à
forward_rnn_25/while/Identity_3IdentityIforward_rnn_25/while/TensorArrayV2Write/TensorListSetItem:output_handle:07^forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp9^forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp0^forward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*
_output_shapes
: 2!
forward_rnn_25/while/Identity_3Ó
forward_rnn_25/while/Identity_4Identity*forward_rnn_25/while/gru_cell_25/add_3:z:07^forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp9^forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp0^forward_rnn_25/while/gru_cell_25/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
forward_rnn_25/while/Identity_4"l
3forward_rnn_25_while_forward_rnn_25_strided_slice_15forward_rnn_25_while_forward_rnn_25_strided_slice_1_0"
Aforward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resourceCforward_rnn_25_while_gru_cell_25_matmul_1_readvariableop_resource_0"
?forward_rnn_25_while_gru_cell_25_matmul_readvariableop_resourceAforward_rnn_25_while_gru_cell_25_matmul_readvariableop_resource_0"v
8forward_rnn_25_while_gru_cell_25_readvariableop_resource:forward_rnn_25_while_gru_cell_25_readvariableop_resource_0"G
forward_rnn_25_while_identity&forward_rnn_25/while/Identity:output:0"K
forward_rnn_25_while_identity_1(forward_rnn_25/while/Identity_1:output:0"K
forward_rnn_25_while_identity_2(forward_rnn_25/while/Identity_2:output:0"K
forward_rnn_25_while_identity_3(forward_rnn_25/while/Identity_3:output:0"K
forward_rnn_25_while_identity_4(forward_rnn_25/while/Identity_4:output:0"ä
oforward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_forward_rnn_25_tensorarrayunstack_tensorlistfromtensorqforward_rnn_25_while_tensorarrayv2read_tensorlistgetitem_forward_rnn_25_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : 2p
6forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp6forward_rnn_25/while/gru_cell_25/MatMul/ReadVariableOp2t
8forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp8forward_rnn_25/while/gru_cell_25/MatMul_1/ReadVariableOp2b
/forward_rnn_25/while/gru_cell_25/ReadVariableOp/forward_rnn_25/while/gru_cell_25/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: "ÌL
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
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:×
Ì-
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
trainable_variables
regularization_losses
	variables
	keras_api
	
signatures
^__call__
*_&call_and_return_all_conditional_losses
`_default_save_signature"¨+
_tf_keras_network+{"name": "Tree_Encoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "Tree_Encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Inputs"}, "name": "Inputs", "inbound_nodes": []}, {"class_name": "Dropout", "config": {"name": "Embedding_Dropout", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "Embedding_Dropout", "inbound_nodes": [[["Inputs", 0, 0, {}]]]}, {"class_name": "Bidirectional", "config": {"name": "Double_Bidirectional_GRU", "trainable": true, "dtype": "float32", "layer": {"class_name": "RNN", "config": {"name": "rnn_25", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "GRUCell", "config": {"name": "gru_cell_25", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}}}, "merge_mode": "concat"}, "name": "Double_Bidirectional_GRU", "inbound_nodes": [[["Embedding_Dropout", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Prediction", "trainable": true, "dtype": "float32", "units": 28, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Prediction", "inbound_nodes": [[["Double_Bidirectional_GRU", 0, 0, {}]]]}], "input_layers": [["Inputs", 0, 0]], "output_layers": [["Prediction", 0, 0]]}, "shared_object_id": 11, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 128]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 128]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, null, 128]}, "float32", "Inputs"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "Tree_Encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Inputs"}, "name": "Inputs", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Dropout", "config": {"name": "Embedding_Dropout", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "Embedding_Dropout", "inbound_nodes": [[["Inputs", 0, 0, {}]]], "shared_object_id": 1}, {"class_name": "Bidirectional", "config": {"name": "Double_Bidirectional_GRU", "trainable": true, "dtype": "float32", "layer": {"class_name": "RNN", "config": {"name": "rnn_25", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "GRUCell", "config": {"name": "gru_cell_25", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 2}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 3}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}, "shared_object_id": 5}}, "shared_object_id": 6}, "merge_mode": "concat"}, "name": "Double_Bidirectional_GRU", "inbound_nodes": [[["Embedding_Dropout", 0, 0, {}]]], "shared_object_id": 7}, {"class_name": "Dense", "config": {"name": "Prediction", "trainable": true, "dtype": "float32", "units": 28, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Prediction", "inbound_nodes": [[["Double_Bidirectional_GRU", 0, 0, {}]]], "shared_object_id": 10}], "input_layers": [["Inputs", 0, 0]], "output_layers": [["Prediction", 0, 0]]}}}
÷"ô
_tf_keras_input_layerÔ{"class_name": "InputLayer", "name": "Inputs", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 128]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Inputs"}}
·

trainable_variables
regularization_losses
	variables
	keras_api
a__call__
*b&call_and_return_all_conditional_losses"¨
_tf_keras_layer{"name": "Embedding_Dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "Embedding_Dropout", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "inbound_nodes": [[["Inputs", 0, 0, {}]]], "shared_object_id": 1}
º
forward_layer
backward_layer
trainable_variables
regularization_losses
	variables
	keras_api
c__call__
*d&call_and_return_all_conditional_losses"
_tf_keras_layerê{"name": "Double_Bidirectional_GRU", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Bidirectional", "config": {"name": "Double_Bidirectional_GRU", "trainable": true, "dtype": "float32", "layer": {"class_name": "RNN", "config": {"name": "rnn_25", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "GRUCell", "config": {"name": "gru_cell_25", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 2}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 3}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}, "shared_object_id": 5}}, "shared_object_id": 6}, "merge_mode": "concat"}, "inbound_nodes": [[["Embedding_Dropout", 0, 0, {}]]], "shared_object_id": 7, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 128]}}
	

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
e__call__
*f&call_and_return_all_conditional_losses"ñ
_tf_keras_layer×{"name": "Prediction", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "Prediction", "trainable": true, "dtype": "float32", "units": 28, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["Double_Bidirectional_GRU", 0, 0, {}]]], "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 13}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
Ê
trainable_variables
 non_trainable_variables
!layer_regularization_losses
regularization_losses
	variables
"metrics

#layers
$layer_metrics
^__call__
`_default_save_signature
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
,
gserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­

trainable_variables
%non_trainable_variables
&layer_regularization_losses
regularization_losses
	variables
'metrics

(layers
)layer_metrics
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
Û
*cell
+
state_spec
,trainable_variables
-regularization_losses
.	variables
/	keras_api
h__call__
*i&call_and_return_all_conditional_losses"²
_tf_keras_rnn_layer{"name": "forward_rnn_25", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "RNN", "config": {"name": "forward_rnn_25", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "GRUCell", "config": {"name": "gru_cell_25", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}, "shared_object_id": 17}}, "shared_object_id": 18, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 128]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 19}]}
Ü
0cell
1
state_spec
2trainable_variables
3regularization_losses
4	variables
5	keras_api
j__call__
*k&call_and_return_all_conditional_losses"³
_tf_keras_rnn_layer{"name": "backward_rnn_25", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "RNN", "config": {"name": "backward_rnn_25", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": true, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "GRUCell", "config": {"name": "gru_cell_25", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 20}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 21}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 22}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}, "shared_object_id": 23}}, "shared_object_id": 24, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 128]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 25}]}
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
­
trainable_variables
6non_trainable_variables
7layer_regularization_losses
regularization_losses
	variables
8metrics

9layers
:layer_metrics
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
$:"	2Prediction/kernel
:2Prediction/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
trainable_variables
;non_trainable_variables
<layer_regularization_losses
regularization_losses
	variables
=metrics

>layers
?layer_metrics
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
N:L
2:Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/kernel
X:V
2DDouble_Bidirectional_GRU/forward_rnn_25/gru_cell_25/recurrent_kernel
K:I	28Double_Bidirectional_GRU/forward_rnn_25/gru_cell_25/bias
O:M
2;Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/kernel
Y:W
2EDouble_Bidirectional_GRU/backward_rnn_25/gru_cell_25/recurrent_kernel
L:J	29Double_Bidirectional_GRU/backward_rnn_25/gru_cell_25/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
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
	

kernel
recurrent_kernel
bias
@trainable_variables
Aregularization_losses
B	variables
C	keras_api
l__call__
*m&call_and_return_all_conditional_losses"È
_tf_keras_layer®{"name": "gru_cell_25", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "GRUCell", "config": {"name": "gru_cell_25", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}, "shared_object_id": 17}
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
¹
,trainable_variables
Dnon_trainable_variables
Elayer_regularization_losses
-regularization_losses
.	variables
Fmetrics

Glayers
Hlayer_metrics

Istates
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
	

kernel
recurrent_kernel
bias
Jtrainable_variables
Kregularization_losses
L	variables
M	keras_api
n__call__
*o&call_and_return_all_conditional_losses"È
_tf_keras_layer®{"name": "gru_cell_25", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "GRUCell", "config": {"name": "gru_cell_25", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 20}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 21}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 22}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}, "shared_object_id": 23}
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
¹
2trainable_variables
Nnon_trainable_variables
Olayer_regularization_losses
3regularization_losses
4	variables
Pmetrics

Qlayers
Rlayer_metrics

Sstates
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
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
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
­
@trainable_variables
Tnon_trainable_variables
Ulayer_regularization_losses
Aregularization_losses
B	variables
Vmetrics

Wlayers
Xlayer_metrics
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
*0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
­
Jtrainable_variables
Ynon_trainable_variables
Zlayer_regularization_losses
Kregularization_losses
L	variables
[metrics

\layers
]layer_metrics
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
00"
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
2
0__inference_Tree_Encoder_layer_call_fn_387380098
0__inference_Tree_Encoder_layer_call_fn_387380654
0__inference_Tree_Encoder_layer_call_fn_387380675
0__inference_Tree_Encoder_layer_call_fn_387380564À
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
ú2÷
K__inference_Tree_Encoder_layer_call_and_return_conditional_losses_387380995
K__inference_Tree_Encoder_layer_call_and_return_conditional_losses_387381322
K__inference_Tree_Encoder_layer_call_and_return_conditional_losses_387380587
K__inference_Tree_Encoder_layer_call_and_return_conditional_losses_387380610À
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
ï2ì
$__inference__wrapped_model_387378557Ã
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
¨2¥
5__inference_Embedding_Dropout_layer_call_fn_387381327
5__inference_Embedding_Dropout_layer_call_fn_387381332´
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
P__inference_Embedding_Dropout_layer_call_and_return_conditional_losses_387381337
P__inference_Embedding_Dropout_layer_call_and_return_conditional_losses_387381349´
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
è2å
<__inference_Double_Bidirectional_GRU_layer_call_fn_387381366
<__inference_Double_Bidirectional_GRU_layer_call_fn_387381383æ
Ý²Ù
FullArgSpecO
argsGD
jself
jinputs

jtraining
jmask
jinitial_state
j	constants
varargs
 
varkw
 
defaults
p 

 

 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
W__inference_Double_Bidirectional_GRU_layer_call_and_return_conditional_losses_387381695
W__inference_Double_Bidirectional_GRU_layer_call_and_return_conditional_losses_387382007æ
Ý²Ù
FullArgSpecO
argsGD
jself
jinputs

jtraining
jmask
jinitial_state
j	constants
varargs
 
varkw
 
defaults
p 

 

 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ø2Õ
.__inference_Prediction_layer_call_fn_387382016¢
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
ó2ð
I__inference_Prediction_layer_call_and_return_conditional_losses_387382027¢
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
ÍBÊ
'__inference_signature_wrapper_387380633Inputs"
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
 
Ô2Ñ
2__inference_forward_rnn_25_layer_call_fn_387382038
2__inference_forward_rnn_25_layer_call_fn_387382049æ
Ý²Ù
FullArgSpecO
argsGD
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults

 
p 

 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
M__inference_forward_rnn_25_layer_call_and_return_conditional_losses_387382205
M__inference_forward_rnn_25_layer_call_and_return_conditional_losses_387382361æ
Ý²Ù
FullArgSpecO
argsGD
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults

 
p 

 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ö2Ó
3__inference_backward_rnn_25_layer_call_fn_387382372
3__inference_backward_rnn_25_layer_call_fn_387382383æ
Ý²Ù
FullArgSpecO
argsGD
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults

 
p 

 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
N__inference_backward_rnn_25_layer_call_and_return_conditional_losses_387382541
N__inference_backward_rnn_25_layer_call_and_return_conditional_losses_387382699æ
Ý²Ù
FullArgSpecO
argsGD
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults

 
p 

 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¦2£
/__inference_gru_cell_25_layer_call_fn_387382713
/__inference_gru_cell_25_layer_call_fn_387382727¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

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
Ü2Ù
J__inference_gru_cell_25_layer_call_and_return_conditional_losses_387382766
J__inference_gru_cell_25_layer_call_and_return_conditional_losses_387382805¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

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
¦2£
/__inference_gru_cell_25_layer_call_fn_387382819
/__inference_gru_cell_25_layer_call_fn_387382833¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

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
Ü2Ù
J__inference_gru_cell_25_layer_call_and_return_conditional_losses_387382872
J__inference_gru_cell_25_layer_call_and_return_conditional_losses_387382911¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

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
 Ú
W__inference_Double_Bidirectional_GRU_layer_call_and_return_conditional_losses_387381695M¢J
C¢@
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 

 

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 Ú
W__inference_Double_Bidirectional_GRU_layer_call_and_return_conditional_losses_387382007M¢J
C¢@
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 

 

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ²
<__inference_Double_Bidirectional_GRU_layer_call_fn_387381366rM¢J
C¢@
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 

 

 
ª "ÿÿÿÿÿÿÿÿÿ²
<__inference_Double_Bidirectional_GRU_layer_call_fn_387381383rM¢J
C¢@
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 

 

 
ª "ÿÿÿÿÿÿÿÿÿÌ
P__inference_Embedding_Dropout_layer_call_and_return_conditional_losses_387381337xA¢>
7¢4
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ì
P__inference_Embedding_Dropout_layer_call_and_return_conditional_losses_387381349xA¢>
7¢4
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¤
5__inference_Embedding_Dropout_layer_call_fn_387381327kA¢>
7¢4
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¤
5__inference_Embedding_Dropout_layer_call_fn_387381332kA¢>
7¢4
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª
I__inference_Prediction_layer_call_and_return_conditional_losses_387382027]0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_Prediction_layer_call_fn_387382016P0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÇ
K__inference_Tree_Encoder_layer_call_and_return_conditional_losses_387380587xE¢B
;¢8
.+
Inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ç
K__inference_Tree_Encoder_layer_call_and_return_conditional_losses_387380610xE¢B
;¢8
.+
Inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ç
K__inference_Tree_Encoder_layer_call_and_return_conditional_losses_387380995xE¢B
;¢8
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ç
K__inference_Tree_Encoder_layer_call_and_return_conditional_losses_387381322xE¢B
;¢8
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
0__inference_Tree_Encoder_layer_call_fn_387380098kE¢B
;¢8
.+
Inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
0__inference_Tree_Encoder_layer_call_fn_387380564kE¢B
;¢8
.+
Inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
0__inference_Tree_Encoder_layer_call_fn_387380654kE¢B
;¢8
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
0__inference_Tree_Encoder_layer_call_fn_387380675kE¢B
;¢8
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ«
$__inference__wrapped_model_387378557=¢:
3¢0
.+
Inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "7ª4
2

Prediction$!

PredictionÿÿÿÿÿÿÿÿÿÖ
N__inference_backward_rnn_25_layer_call_and_return_conditional_losses_387382541T¢Q
J¢G
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 Ö
N__inference_backward_rnn_25_layer_call_and_return_conditional_losses_387382699T¢Q
J¢G
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ­
3__inference_backward_rnn_25_layer_call_fn_387382372vT¢Q
J¢G
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 

 
ª "ÿÿÿÿÿÿÿÿÿ­
3__inference_backward_rnn_25_layer_call_fn_387382383vT¢Q
J¢G
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 

 
ª "ÿÿÿÿÿÿÿÿÿÕ
M__inference_forward_rnn_25_layer_call_and_return_conditional_losses_387382205T¢Q
J¢G
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 Õ
M__inference_forward_rnn_25_layer_call_and_return_conditional_losses_387382361T¢Q
J¢G
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ¬
2__inference_forward_rnn_25_layer_call_fn_387382038vT¢Q
J¢G
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 

 
ª "ÿÿÿÿÿÿÿÿÿ¬
2__inference_forward_rnn_25_layer_call_fn_387382049vT¢Q
J¢G
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 

 
ª "ÿÿÿÿÿÿÿÿÿ
J__inference_gru_cell_25_layer_call_and_return_conditional_losses_387382766»^¢[
T¢Q
!
inputsÿÿÿÿÿÿÿÿÿ
(¢%
# 
states/0ÿÿÿÿÿÿÿÿÿ
p 
ª "T¢Q
J¢G

0/0ÿÿÿÿÿÿÿÿÿ
%"
 
0/1/0ÿÿÿÿÿÿÿÿÿ
 
J__inference_gru_cell_25_layer_call_and_return_conditional_losses_387382805»^¢[
T¢Q
!
inputsÿÿÿÿÿÿÿÿÿ
(¢%
# 
states/0ÿÿÿÿÿÿÿÿÿ
p
ª "T¢Q
J¢G

0/0ÿÿÿÿÿÿÿÿÿ
%"
 
0/1/0ÿÿÿÿÿÿÿÿÿ
 
J__inference_gru_cell_25_layer_call_and_return_conditional_losses_387382872»^¢[
T¢Q
!
inputsÿÿÿÿÿÿÿÿÿ
(¢%
# 
states/0ÿÿÿÿÿÿÿÿÿ
p 
ª "T¢Q
J¢G

0/0ÿÿÿÿÿÿÿÿÿ
%"
 
0/1/0ÿÿÿÿÿÿÿÿÿ
 
J__inference_gru_cell_25_layer_call_and_return_conditional_losses_387382911»^¢[
T¢Q
!
inputsÿÿÿÿÿÿÿÿÿ
(¢%
# 
states/0ÿÿÿÿÿÿÿÿÿ
p
ª "T¢Q
J¢G

0/0ÿÿÿÿÿÿÿÿÿ
%"
 
0/1/0ÿÿÿÿÿÿÿÿÿ
 á
/__inference_gru_cell_25_layer_call_fn_387382713­^¢[
T¢Q
!
inputsÿÿÿÿÿÿÿÿÿ
(¢%
# 
states/0ÿÿÿÿÿÿÿÿÿ
p 
ª "F¢C

0ÿÿÿÿÿÿÿÿÿ
# 

1/0ÿÿÿÿÿÿÿÿÿá
/__inference_gru_cell_25_layer_call_fn_387382727­^¢[
T¢Q
!
inputsÿÿÿÿÿÿÿÿÿ
(¢%
# 
states/0ÿÿÿÿÿÿÿÿÿ
p
ª "F¢C

0ÿÿÿÿÿÿÿÿÿ
# 

1/0ÿÿÿÿÿÿÿÿÿá
/__inference_gru_cell_25_layer_call_fn_387382819­^¢[
T¢Q
!
inputsÿÿÿÿÿÿÿÿÿ
(¢%
# 
states/0ÿÿÿÿÿÿÿÿÿ
p 
ª "F¢C

0ÿÿÿÿÿÿÿÿÿ
# 

1/0ÿÿÿÿÿÿÿÿÿá
/__inference_gru_cell_25_layer_call_fn_387382833­^¢[
T¢Q
!
inputsÿÿÿÿÿÿÿÿÿ
(¢%
# 
states/0ÿÿÿÿÿÿÿÿÿ
p
ª "F¢C

0ÿÿÿÿÿÿÿÿÿ
# 

1/0ÿÿÿÿÿÿÿÿÿ¸
'__inference_signature_wrapper_387380633G¢D
¢ 
=ª:
8
Inputs.+
Inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"7ª4
2

Prediction$!

Predictionÿÿÿÿÿÿÿÿÿ