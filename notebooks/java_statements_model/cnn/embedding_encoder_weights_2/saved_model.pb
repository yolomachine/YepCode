߽
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
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
?
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
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
delete_old_dirsbool(?
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
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype?
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
?
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
executor_typestring ?
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
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718??
|
dense_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_28/kernel
u
#dense_28/kernel/Read/ReadVariableOpReadVariableOpdense_28/kernel* 
_output_shapes
:
??*
dtype0
s
dense_28/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_28/bias
l
!dense_28/bias/Read/ReadVariableOpReadVariableOpdense_28/bias*
_output_shapes	
:?*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
layer-0
layer_with_weights-0
layer-1
trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
h

kernel
	bias

trainable_variables
regularization_losses
	variables
	keras_api

0
	1
 

0
	1
?
trainable_variables
non_trainable_variables
layer_regularization_losses
regularization_losses
	variables
metrics

layers
layer_metrics
 
[Y
VARIABLE_VALUEdense_28/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_28/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
	1
 

0
	1
?

trainable_variables
non_trainable_variables
layer_regularization_losses
regularization_losses
	variables
metrics

layers
layer_metrics
 
 
 

0
1
 
 
 
 
 
 
?
serving_default_input_29Placeholder*B
_output_shapes0
.:,????????????????????????????*
dtype0*7
shape.:,????????????????????????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_29dense_28/kerneldense_28/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *1
f,R*
(__inference_signature_wrapper_1294347430
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_28/kernel/Read/ReadVariableOp!dense_28/bias/Read/ReadVariableOpConst*
Tin
2*
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
GPU2*0J 8? *,
f'R%
#__inference__traced_save_1294347576
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_28/kerneldense_28/bias*
Tin
2*
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
GPU2*0J 8? */
f*R(
&__inference__traced_restore_1294347592??
?
?
&__inference__traced_restore_1294347592
file_prefix4
 assignvariableop_dense_28_kernel:
??/
 assignvariableop_1_dense_28_bias:	?

identity_3??AssignVariableOp?AssignVariableOp_1?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0* 
_output_shapes
:::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_dense_28_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_28_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_2Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_2?

Identity_3IdentityIdentity_2:output:0^AssignVariableOp^AssignVariableOp_1*
T0*
_output_shapes
: 2

Identity_3"!

identity_3Identity_3:output:0*
_input_shapes
: : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?'
?
Q__inference_Embedding_Encoder_layer_call_and_return_conditional_losses_1294347478

inputs>
*dense_28_tensordot_readvariableop_resource:
??7
(dense_28_biasadd_readvariableop_resource:	?
identity??dense_28/BiasAdd/ReadVariableOp?!dense_28/Tensordot/ReadVariableOp?
!dense_28/Tensordot/ReadVariableOpReadVariableOp*dense_28_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!dense_28/Tensordot/ReadVariableOp|
dense_28/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_28/Tensordot/axes?
dense_28/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
dense_28/Tensordot/freej
dense_28/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_28/Tensordot/Shape?
 dense_28/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_28/Tensordot/GatherV2/axis?
dense_28/Tensordot/GatherV2GatherV2!dense_28/Tensordot/Shape:output:0 dense_28/Tensordot/free:output:0)dense_28/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_28/Tensordot/GatherV2?
"dense_28/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_28/Tensordot/GatherV2_1/axis?
dense_28/Tensordot/GatherV2_1GatherV2!dense_28/Tensordot/Shape:output:0 dense_28/Tensordot/axes:output:0+dense_28/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_28/Tensordot/GatherV2_1~
dense_28/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_28/Tensordot/Const?
dense_28/Tensordot/ProdProd$dense_28/Tensordot/GatherV2:output:0!dense_28/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_28/Tensordot/Prod?
dense_28/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_28/Tensordot/Const_1?
dense_28/Tensordot/Prod_1Prod&dense_28/Tensordot/GatherV2_1:output:0#dense_28/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_28/Tensordot/Prod_1?
dense_28/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_28/Tensordot/concat/axis?
dense_28/Tensordot/concatConcatV2 dense_28/Tensordot/free:output:0 dense_28/Tensordot/axes:output:0'dense_28/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_28/Tensordot/concat?
dense_28/Tensordot/stackPack dense_28/Tensordot/Prod:output:0"dense_28/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_28/Tensordot/stack?
dense_28/Tensordot/transpose	Transposeinputs"dense_28/Tensordot/concat:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
dense_28/Tensordot/transpose?
dense_28/Tensordot/ReshapeReshape dense_28/Tensordot/transpose:y:0!dense_28/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_28/Tensordot/Reshape?
dense_28/Tensordot/MatMulMatMul#dense_28/Tensordot/Reshape:output:0)dense_28/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_28/Tensordot/MatMul?
dense_28/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_28/Tensordot/Const_2?
 dense_28/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_28/Tensordot/concat_1/axis?
dense_28/Tensordot/concat_1ConcatV2$dense_28/Tensordot/GatherV2:output:0#dense_28/Tensordot/Const_2:output:0)dense_28/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_28/Tensordot/concat_1?
dense_28/TensordotReshape#dense_28/Tensordot/MatMul:product:0$dense_28/Tensordot/concat_1:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
dense_28/Tensordot?
dense_28/BiasAdd/ReadVariableOpReadVariableOp(dense_28_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_28/BiasAdd/ReadVariableOp?
dense_28/BiasAddBiasAdddense_28/Tensordot:output:0'dense_28/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2
dense_28/BiasAdd?
IdentityIdentitydense_28/BiasAdd:output:0 ^dense_28/BiasAdd/ReadVariableOp"^dense_28/Tensordot/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 2B
dense_28/BiasAdd/ReadVariableOpdense_28/BiasAdd/ReadVariableOp2F
!dense_28/Tensordot/ReadVariableOp!dense_28/Tensordot/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?	
?
Q__inference_Embedding_Encoder_layer_call_and_return_conditional_losses_1294347385

inputs'
dense_28_1294347379:
??"
dense_28_1294347381:	?
identity?? dense_28/StatefulPartitionedCall?
 dense_28/StatefulPartitionedCallStatefulPartitionedCallinputsdense_28_1294347379dense_28_1294347381*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dense_28_layer_call_and_return_conditional_losses_12943473412"
 dense_28/StatefulPartitionedCall?
IdentityIdentity)dense_28/StatefulPartitionedCall:output:0!^dense_28/StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
6__inference_Embedding_Encoder_layer_call_fn_1294347355
input_29
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_29unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_Embedding_Encoder_layer_call_and_return_conditional_losses_12943473482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:l h
B
_output_shapes0
.:,????????????????????????????
"
_user_specified_name
input_29
?3
?
%__inference__wrapped_model_1294347304
input_29P
<embedding_encoder_dense_28_tensordot_readvariableop_resource:
??I
:embedding_encoder_dense_28_biasadd_readvariableop_resource:	?
identity??1Embedding_Encoder/dense_28/BiasAdd/ReadVariableOp?3Embedding_Encoder/dense_28/Tensordot/ReadVariableOp?
3Embedding_Encoder/dense_28/Tensordot/ReadVariableOpReadVariableOp<embedding_encoder_dense_28_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype025
3Embedding_Encoder/dense_28/Tensordot/ReadVariableOp?
)Embedding_Encoder/dense_28/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2+
)Embedding_Encoder/dense_28/Tensordot/axes?
)Embedding_Encoder/dense_28/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)Embedding_Encoder/dense_28/Tensordot/free?
*Embedding_Encoder/dense_28/Tensordot/ShapeShapeinput_29*
T0*
_output_shapes
:2,
*Embedding_Encoder/dense_28/Tensordot/Shape?
2Embedding_Encoder/dense_28/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 24
2Embedding_Encoder/dense_28/Tensordot/GatherV2/axis?
-Embedding_Encoder/dense_28/Tensordot/GatherV2GatherV23Embedding_Encoder/dense_28/Tensordot/Shape:output:02Embedding_Encoder/dense_28/Tensordot/free:output:0;Embedding_Encoder/dense_28/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2/
-Embedding_Encoder/dense_28/Tensordot/GatherV2?
4Embedding_Encoder/dense_28/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 26
4Embedding_Encoder/dense_28/Tensordot/GatherV2_1/axis?
/Embedding_Encoder/dense_28/Tensordot/GatherV2_1GatherV23Embedding_Encoder/dense_28/Tensordot/Shape:output:02Embedding_Encoder/dense_28/Tensordot/axes:output:0=Embedding_Encoder/dense_28/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:21
/Embedding_Encoder/dense_28/Tensordot/GatherV2_1?
*Embedding_Encoder/dense_28/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*Embedding_Encoder/dense_28/Tensordot/Const?
)Embedding_Encoder/dense_28/Tensordot/ProdProd6Embedding_Encoder/dense_28/Tensordot/GatherV2:output:03Embedding_Encoder/dense_28/Tensordot/Const:output:0*
T0*
_output_shapes
: 2+
)Embedding_Encoder/dense_28/Tensordot/Prod?
,Embedding_Encoder/dense_28/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,Embedding_Encoder/dense_28/Tensordot/Const_1?
+Embedding_Encoder/dense_28/Tensordot/Prod_1Prod8Embedding_Encoder/dense_28/Tensordot/GatherV2_1:output:05Embedding_Encoder/dense_28/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2-
+Embedding_Encoder/dense_28/Tensordot/Prod_1?
0Embedding_Encoder/dense_28/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0Embedding_Encoder/dense_28/Tensordot/concat/axis?
+Embedding_Encoder/dense_28/Tensordot/concatConcatV22Embedding_Encoder/dense_28/Tensordot/free:output:02Embedding_Encoder/dense_28/Tensordot/axes:output:09Embedding_Encoder/dense_28/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2-
+Embedding_Encoder/dense_28/Tensordot/concat?
*Embedding_Encoder/dense_28/Tensordot/stackPack2Embedding_Encoder/dense_28/Tensordot/Prod:output:04Embedding_Encoder/dense_28/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2,
*Embedding_Encoder/dense_28/Tensordot/stack?
.Embedding_Encoder/dense_28/Tensordot/transpose	Transposeinput_294Embedding_Encoder/dense_28/Tensordot/concat:output:0*
T0*B
_output_shapes0
.:,????????????????????????????20
.Embedding_Encoder/dense_28/Tensordot/transpose?
,Embedding_Encoder/dense_28/Tensordot/ReshapeReshape2Embedding_Encoder/dense_28/Tensordot/transpose:y:03Embedding_Encoder/dense_28/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2.
,Embedding_Encoder/dense_28/Tensordot/Reshape?
+Embedding_Encoder/dense_28/Tensordot/MatMulMatMul5Embedding_Encoder/dense_28/Tensordot/Reshape:output:0;Embedding_Encoder/dense_28/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2-
+Embedding_Encoder/dense_28/Tensordot/MatMul?
,Embedding_Encoder/dense_28/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2.
,Embedding_Encoder/dense_28/Tensordot/Const_2?
2Embedding_Encoder/dense_28/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 24
2Embedding_Encoder/dense_28/Tensordot/concat_1/axis?
-Embedding_Encoder/dense_28/Tensordot/concat_1ConcatV26Embedding_Encoder/dense_28/Tensordot/GatherV2:output:05Embedding_Encoder/dense_28/Tensordot/Const_2:output:0;Embedding_Encoder/dense_28/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2/
-Embedding_Encoder/dense_28/Tensordot/concat_1?
$Embedding_Encoder/dense_28/TensordotReshape5Embedding_Encoder/dense_28/Tensordot/MatMul:product:06Embedding_Encoder/dense_28/Tensordot/concat_1:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2&
$Embedding_Encoder/dense_28/Tensordot?
1Embedding_Encoder/dense_28/BiasAdd/ReadVariableOpReadVariableOp:embedding_encoder_dense_28_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype023
1Embedding_Encoder/dense_28/BiasAdd/ReadVariableOp?
"Embedding_Encoder/dense_28/BiasAddBiasAdd-Embedding_Encoder/dense_28/Tensordot:output:09Embedding_Encoder/dense_28/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2$
"Embedding_Encoder/dense_28/BiasAdd?
IdentityIdentity+Embedding_Encoder/dense_28/BiasAdd:output:02^Embedding_Encoder/dense_28/BiasAdd/ReadVariableOp4^Embedding_Encoder/dense_28/Tensordot/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 2f
1Embedding_Encoder/dense_28/BiasAdd/ReadVariableOp1Embedding_Encoder/dense_28/BiasAdd/ReadVariableOp2j
3Embedding_Encoder/dense_28/Tensordot/ReadVariableOp3Embedding_Encoder/dense_28/Tensordot/ReadVariableOp:l h
B
_output_shapes0
.:,????????????????????????????
"
_user_specified_name
input_29
?	
?
Q__inference_Embedding_Encoder_layer_call_and_return_conditional_losses_1294347410
input_29'
dense_28_1294347404:
??"
dense_28_1294347406:	?
identity?? dense_28/StatefulPartitionedCall?
 dense_28/StatefulPartitionedCallStatefulPartitionedCallinput_29dense_28_1294347404dense_28_1294347406*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dense_28_layer_call_and_return_conditional_losses_12943473412"
 dense_28/StatefulPartitionedCall?
IdentityIdentity)dense_28/StatefulPartitionedCall:output:0!^dense_28/StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall:l h
B
_output_shapes0
.:,????????????????????????????
"
_user_specified_name
input_29
?
?
-__inference_dense_28_layer_call_fn_1294347517

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dense_28_layer_call_and_return_conditional_losses_12943473412
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
6__inference_Embedding_Encoder_layer_call_fn_1294347448

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_Embedding_Encoder_layer_call_and_return_conditional_losses_12943473852
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?	
?
Q__inference_Embedding_Encoder_layer_call_and_return_conditional_losses_1294347419
input_29'
dense_28_1294347413:
??"
dense_28_1294347415:	?
identity?? dense_28/StatefulPartitionedCall?
 dense_28/StatefulPartitionedCallStatefulPartitionedCallinput_29dense_28_1294347413dense_28_1294347415*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dense_28_layer_call_and_return_conditional_losses_12943473412"
 dense_28/StatefulPartitionedCall?
IdentityIdentity)dense_28/StatefulPartitionedCall:output:0!^dense_28/StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall:l h
B
_output_shapes0
.:,????????????????????????????
"
_user_specified_name
input_29
?
?
#__inference__traced_save_1294347576
file_prefix.
*savev2_dense_28_kernel_read_readvariableop,
(savev2_dense_28_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
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
Const_1?
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
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_28_kernel_read_readvariableop(savev2_dense_28_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
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

identity_1Identity_1:output:0**
_input_shapes
: :
??:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:

_output_shapes
: 
?	
?
Q__inference_Embedding_Encoder_layer_call_and_return_conditional_losses_1294347348

inputs'
dense_28_1294347342:
??"
dense_28_1294347344:	?
identity?? dense_28/StatefulPartitionedCall?
 dense_28/StatefulPartitionedCallStatefulPartitionedCallinputsdense_28_1294347342dense_28_1294347344*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dense_28_layer_call_and_return_conditional_losses_12943473412"
 dense_28/StatefulPartitionedCall?
IdentityIdentity)dense_28/StatefulPartitionedCall:output:0!^dense_28/StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
(__inference_signature_wrapper_1294347430
input_29
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_29unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *.
f)R'
%__inference__wrapped_model_12943473042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:l h
B
_output_shapes0
.:,????????????????????????????
"
_user_specified_name
input_29
?!
?
H__inference_dense_28_layer_call_and_return_conditional_losses_1294347341

inputs5
!tensordot_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesu
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
6__inference_Embedding_Encoder_layer_call_fn_1294347401
input_29
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_29unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_Embedding_Encoder_layer_call_and_return_conditional_losses_12943473852
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:l h
B
_output_shapes0
.:,????????????????????????????
"
_user_specified_name
input_29
?'
?
Q__inference_Embedding_Encoder_layer_call_and_return_conditional_losses_1294347508

inputs>
*dense_28_tensordot_readvariableop_resource:
??7
(dense_28_biasadd_readvariableop_resource:	?
identity??dense_28/BiasAdd/ReadVariableOp?!dense_28/Tensordot/ReadVariableOp?
!dense_28/Tensordot/ReadVariableOpReadVariableOp*dense_28_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!dense_28/Tensordot/ReadVariableOp|
dense_28/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_28/Tensordot/axes?
dense_28/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
dense_28/Tensordot/freej
dense_28/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_28/Tensordot/Shape?
 dense_28/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_28/Tensordot/GatherV2/axis?
dense_28/Tensordot/GatherV2GatherV2!dense_28/Tensordot/Shape:output:0 dense_28/Tensordot/free:output:0)dense_28/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_28/Tensordot/GatherV2?
"dense_28/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_28/Tensordot/GatherV2_1/axis?
dense_28/Tensordot/GatherV2_1GatherV2!dense_28/Tensordot/Shape:output:0 dense_28/Tensordot/axes:output:0+dense_28/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_28/Tensordot/GatherV2_1~
dense_28/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_28/Tensordot/Const?
dense_28/Tensordot/ProdProd$dense_28/Tensordot/GatherV2:output:0!dense_28/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_28/Tensordot/Prod?
dense_28/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_28/Tensordot/Const_1?
dense_28/Tensordot/Prod_1Prod&dense_28/Tensordot/GatherV2_1:output:0#dense_28/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_28/Tensordot/Prod_1?
dense_28/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_28/Tensordot/concat/axis?
dense_28/Tensordot/concatConcatV2 dense_28/Tensordot/free:output:0 dense_28/Tensordot/axes:output:0'dense_28/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_28/Tensordot/concat?
dense_28/Tensordot/stackPack dense_28/Tensordot/Prod:output:0"dense_28/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_28/Tensordot/stack?
dense_28/Tensordot/transpose	Transposeinputs"dense_28/Tensordot/concat:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
dense_28/Tensordot/transpose?
dense_28/Tensordot/ReshapeReshape dense_28/Tensordot/transpose:y:0!dense_28/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_28/Tensordot/Reshape?
dense_28/Tensordot/MatMulMatMul#dense_28/Tensordot/Reshape:output:0)dense_28/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_28/Tensordot/MatMul?
dense_28/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_28/Tensordot/Const_2?
 dense_28/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_28/Tensordot/concat_1/axis?
dense_28/Tensordot/concat_1ConcatV2$dense_28/Tensordot/GatherV2:output:0#dense_28/Tensordot/Const_2:output:0)dense_28/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_28/Tensordot/concat_1?
dense_28/TensordotReshape#dense_28/Tensordot/MatMul:product:0$dense_28/Tensordot/concat_1:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
dense_28/Tensordot?
dense_28/BiasAdd/ReadVariableOpReadVariableOp(dense_28_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_28/BiasAdd/ReadVariableOp?
dense_28/BiasAddBiasAdddense_28/Tensordot:output:0'dense_28/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2
dense_28/BiasAdd?
IdentityIdentitydense_28/BiasAdd:output:0 ^dense_28/BiasAdd/ReadVariableOp"^dense_28/Tensordot/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 2B
dense_28/BiasAdd/ReadVariableOpdense_28/BiasAdd/ReadVariableOp2F
!dense_28/Tensordot/ReadVariableOp!dense_28/Tensordot/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
6__inference_Embedding_Encoder_layer_call_fn_1294347439

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_Embedding_Encoder_layer_call_and_return_conditional_losses_12943473482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?!
?
H__inference_dense_28_layer_call_and_return_conditional_losses_1294347547

inputs5
!tensordot_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesu
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
X
input_29L
serving_default_input_29:0,????????????????????????????W
dense_28K
StatefulPartitionedCall:0,????????????????????????????tensorflow/serving/predict:?P
?
layer-0
layer_with_weights-0
layer-1
trainable_variables
regularization_losses
	variables
	keras_api

signatures
__call__
*&call_and_return_all_conditional_losses
_default_save_signature"?
_tf_keras_network?{"name": "Embedding_Encoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "Embedding_Encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 192]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_29"}, "name": "input_29", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_28", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_28", "inbound_nodes": [[["input_29", 0, 0, {}]]]}], "input_layers": [["input_29", 0, 0]], "output_layers": [["dense_28", 0, 0]]}, "shared_object_id": 4, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, null, 192]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 192]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, null, null, 192]}, "float32", "input_29"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "Embedding_Encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 192]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_29"}, "name": "input_29", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Dense", "config": {"name": "dense_28", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_28", "inbound_nodes": [[["input_29", 0, 0, {}]]], "shared_object_id": 3}], "input_layers": [["input_29", 0, 0]], "output_layers": [["dense_28", 0, 0]]}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_29", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 192]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 192]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_29"}}
?	

kernel
	bias

trainable_variables
regularization_losses
	variables
	keras_api
__call__
*&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_28", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_28", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_29", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 192}}, "shared_object_id": 6}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 192]}}
.
0
	1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
?
trainable_variables
non_trainable_variables
layer_regularization_losses
regularization_losses
	variables
metrics

layers
layer_metrics
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
,
serving_default"
signature_map
#:!
??2dense_28/kernel
:?2dense_28/bias
.
0
	1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
?

trainable_variables
non_trainable_variables
layer_regularization_losses
regularization_losses
	variables
metrics

layers
layer_metrics
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
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
?2?
6__inference_Embedding_Encoder_layer_call_fn_1294347355
6__inference_Embedding_Encoder_layer_call_fn_1294347439
6__inference_Embedding_Encoder_layer_call_fn_1294347448
6__inference_Embedding_Encoder_layer_call_fn_1294347401?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
Q__inference_Embedding_Encoder_layer_call_and_return_conditional_losses_1294347478
Q__inference_Embedding_Encoder_layer_call_and_return_conditional_losses_1294347508
Q__inference_Embedding_Encoder_layer_call_and_return_conditional_losses_1294347410
Q__inference_Embedding_Encoder_layer_call_and_return_conditional_losses_1294347419?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
%__inference__wrapped_model_1294347304?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *B??
=?:
input_29,????????????????????????????
?2?
-__inference_dense_28_layer_call_fn_1294347517?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_dense_28_layer_call_and_return_conditional_losses_1294347547?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
(__inference_signature_wrapper_1294347430input_29"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
Q__inference_Embedding_Encoder_layer_call_and_return_conditional_losses_1294347410?	T?Q
J?G
=?:
input_29,????????????????????????????
p 

 
? "@?=
6?3
0,????????????????????????????
? ?
Q__inference_Embedding_Encoder_layer_call_and_return_conditional_losses_1294347419?	T?Q
J?G
=?:
input_29,????????????????????????????
p

 
? "@?=
6?3
0,????????????????????????????
? ?
Q__inference_Embedding_Encoder_layer_call_and_return_conditional_losses_1294347478?	R?O
H?E
;?8
inputs,????????????????????????????
p 

 
? "@?=
6?3
0,????????????????????????????
? ?
Q__inference_Embedding_Encoder_layer_call_and_return_conditional_losses_1294347508?	R?O
H?E
;?8
inputs,????????????????????????????
p

 
? "@?=
6?3
0,????????????????????????????
? ?
6__inference_Embedding_Encoder_layer_call_fn_1294347355?	T?Q
J?G
=?:
input_29,????????????????????????????
p 

 
? "3?0,?????????????????????????????
6__inference_Embedding_Encoder_layer_call_fn_1294347401?	T?Q
J?G
=?:
input_29,????????????????????????????
p

 
? "3?0,?????????????????????????????
6__inference_Embedding_Encoder_layer_call_fn_1294347439?	R?O
H?E
;?8
inputs,????????????????????????????
p 

 
? "3?0,?????????????????????????????
6__inference_Embedding_Encoder_layer_call_fn_1294347448?	R?O
H?E
;?8
inputs,????????????????????????????
p

 
? "3?0,?????????????????????????????
%__inference__wrapped_model_1294347304?	L?I
B??
=?:
input_29,????????????????????????????
? "N?K
I
dense_28=?:
dense_28,?????????????????????????????
H__inference_dense_28_layer_call_and_return_conditional_losses_1294347547?	J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
-__inference_dense_28_layer_call_fn_1294347517?	J?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
(__inference_signature_wrapper_1294347430?	X?U
? 
N?K
I
input_29=?:
input_29,????????????????????????????"N?K
I
dense_28=?:
dense_28,????????????????????????????