# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing general type definitions used across vectorbtpro.

This module provides foundational type aliases, protocols, and utilities for common data structures
such as sequences, scalars, arrays, and datetime representations within vectorbtpro.
"""

import ast
import sys
from datetime import date, datetime, time, timedelta, tzinfo
from enum import EnumMeta
from pathlib import Path

if sys.version_info < (3, 9):
    import typing

    typing.__all__.append("TextIO")
from typing import *

import numpy as np
from mypy_extensions import KwArg, VarArg
from pandas import (
    DataFrame as Frame,
)
from pandas import (
    DatetimeIndex,
    Index,
    IndexSlice,
    MultiIndex,
    PeriodIndex,
    Series,
    Timestamp,
)
from pandas import (
    Timedelta as PandasTimedelta,
)
from pandas.core.groupby import GroupBy as PandasGroupBy
from pandas.core.resample import Resampler as PandasResampler
from pandas.tseries.offsets import BaseOffset

if TYPE_CHECKING:
    from plotly.basedatatypes import BaseFigure, BaseTraceType
    from plotly.graph_objects import Figure, FigureWidget
else:
    Figure = "plotly.graph_objects.Figure"
    FigureWidget = "plotly.graph_objects.FigureWidget"
    BaseFigure = "plotly.basedatatypes.BaseFigure"
    BaseTraceType = "plotly.basedatatypes.BaseTraceType"
try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol
try:
    from typing import Self
except ImportError:
    pass

if TYPE_CHECKING:
    from vectorbtpro.base.grouping.base import Grouper
    from vectorbtpro.base.indexing import hslice
    from vectorbtpro.base.resampling.base import Resampler
    from vectorbtpro.base.wrapping import ArrayWrapper
    from vectorbtpro.data.base import Data
    from vectorbtpro.generic.splitting.base import FixRange, RelRange
    from vectorbtpro.indicators.factory import IndicatorBase
    from vectorbtpro.portfolio.enums import (
        FlexOrderContext,
        GroupContext,
        Order,
        OrderContext,
        PostOrderContext,
        PostSignalContext,
        RowContext,
        SegmentContext,
        SignalContext,
        SignalSegmentContext,
        SimulationContext,
    )
    from vectorbtpro.utils.chunking import (
        ChunkMeta,
        ChunkMetaGenerator,
        ChunkTaker,
        NotChunked,
        Sizer,
    )
    from vectorbtpro.utils.datetime_ import DTC, DTCNT
    from vectorbtpro.utils.execution import ExecutionEngine, Task
    from vectorbtpro.utils.jitting import Jitter
    from vectorbtpro.utils.knowledge.asset_pipelines import AssetPipeline
    from vectorbtpro.utils.knowledge.base_asset_funcs import AssetFunc
    from vectorbtpro.utils.knowledge.base_assets import KnowledgeAsset
    from vectorbtpro.utils.knowledge.chatting import (
        Completions,
        EmbeddedDocument,
        Embeddings,
        ObjectStore,
        ScoredDocument,
        StoreDocument,
        TextSplitter,
        Tokenizer,
    )
    from vectorbtpro.utils.knowledge.custom_assets import (
        ExamplesAsset,
        MessagesAsset,
        PagesAsset,
        VBTAsset,
    )
    from vectorbtpro.utils.knowledge.formatting import ContentFormatter
    from vectorbtpro.utils.merging import MergeFunc
    from vectorbtpro.utils.parsing import Regex
    from vectorbtpro.utils.selection import LabelSel, PosSel
    from vectorbtpro.utils.template import CustomTemplate
else:
    Regex = "Regex"
    Task = "Task"
    ExecutionEngine = "ExecutionEngine"
    Sizer = "Sizer"
    NotChunked = "NotChunked"
    ChunkTaker = "ChunkTaker"
    ChunkMeta = "ChunkMeta"
    ChunkMetaGenerator = "ChunkMetaGenerator"
    TraceUpdater = "TraceUpdater"
    Jitter = "Jitter"
    CustomTemplate = "CustomTemplate"
    DTC = "DTC"
    DTCNT = "DTCNT"
    PosSel = "PosSel"
    LabelSel = "LabelSel"
    MergeFunc = "MergeFunc"
    AssetFunc = "AssetFunc"
    AssetPipeline = "AssetPipeline"
    KnowledgeAsset = "KnowledgeAsset"
    Tokenizer = "Tokenizer"
    Embeddings = "Embeddings"
    Completions = "Completions"
    TextSplitter = "TextSplitter"
    StoreDocument = "StoreDocument"
    ObjectStore = "ObjectStore"
    EmbeddedDocument = "EmbeddedDocument"
    ScoredDocument = "ScoredDocument"
    VBTAsset = "VBTAsset"
    PagesAsset = "PagesAsset"
    MessagesAsset = "MessagesAsset"
    ExamplesAsset = "ExamplesAsset"
    ContentFormatter = "ContentFormatter"
    hslice = "hslice"
    Grouper = "Grouper"
    Resampler = "Resampler"
    ArrayWrapper = "ArrayWrapper"
    Data = "Data"
    FixRange = "FixRange"
    RelRange = "RelRange"
    IndicatorBase = "IndicatorBase"
    SignalContext = "SignalContext"
    PostSignalContext = "PostSignalContext"
    SignalSegmentContext = "SignalSegmentContext"
    SimulationContext = "SimulationContext"
    GroupContext = "GroupContext"
    RowContext = "RowContext"
    SegmentContext = "SegmentContext"
    OrderContext = "OrderContext"
    FlexOrderContext = "FlexOrderContext"
    PostOrderContext = "PostOrderContext"
    Order = "Order"

__all__ = []

# Generic types
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])
MaybeType = Union[T, Type[T]]

# Scalars
Scalar = Union[str, float, int, complex, bool, object, np.generic]
Number = Union[int, float, complex, np.number, np.bool_]
Int = Union[int, np.integer]
Float = Union[float, np.floating]
IntFloat = Union[Int, Float]
IntStr = Union[Int, str]

# Basic sequences
MaybeTuple = Union[T, Tuple[T, ...]]
MaybeList = Union[T, List[T]]
MaybeSet = Union[T, Set[T]]
TupleList = Union[List[T], Tuple[T, ...]]
MaybeTupleList = Union[T, List[T], Tuple[T, ...]]
MaybeIterable = Union[T, Iterable[T]]
MaybeSequence = Union[T, Sequence[T]]
MaybeDict = Union[Dict[Hashable, T], T]
MappingSequence = Union[Mapping[Hashable, T], Sequence[T]]
MaybeMappingSequence = Union[T, Mapping[Hashable, T], Sequence[T]]
SetLike = Union[None, Set[T]]
Items = Iterator[Tuple[Hashable, Any]]


# Arrays
class SupportsArrayT(Protocol):
    def __array__(self) -> np.ndarray: ...


DTypeLike = Any
PandasDTypeLike = Any
TypeLike = MaybeIterable[Union[Type, str, Regex]]
Shape = Tuple[int, ...]
ShapeLike = Union[int, Shape]
Array = np.ndarray  # ready to be used for n-dim data
Array1d = np.ndarray
Array2d = np.ndarray
Array3d = np.ndarray
Record = np.void
RecordArray = np.ndarray
RecordArray2d = np.ndarray
RecArray = np.recarray
MaybeArray = Union[Scalar, Array]
MaybeIndexArray = Union[int, slice, Array1d, Array2d]
SeriesFrame = Union[Series, Frame]
MaybeSeries = Union[Scalar, Series]
MaybeSeriesFrame = Union[T, Series, Frame]
PandasArray = Union[Index, Series, Frame]
AnyArray = Union[Array, PandasArray]
AnyArray1d = Union[Array1d, Index, Series]
AnyArray2d = Union[Array2d, Frame]
ArrayLike = Union[Scalar, Sequence[Scalar], Sequence[Sequence[Any]], SupportsArrayT, Array]
IndexLike = Union[range, Sequence[Scalar], SupportsArrayT]
FlexArray1d = Array1d
FlexArray2d = Array2d
FlexArray1dLike = Union[Scalar, Array1d, Array2d]
FlexArray2dLike = Union[Scalar, Array1d, Array2d]
IndexFromLike = Union[None, str, int, Any]

# Templates
CustomTemplateLike = Union[str, Callable, CustomTemplate]

# Labels
Label = Hashable
Labels = Sequence[Label]
Level = Union[str, int]
LevelSequence = Sequence[Level]
MaybeLevelSequence = Union[Level, LevelSequence]

# Datetime
Datetime = Union[Timestamp, np.datetime64, datetime]
DatetimeLike = Union[str, int, float, Datetime]
Timedelta = Union[PandasTimedelta, np.timedelta64, timedelta]
TimedeltaLike = Union[str, int, float, Timedelta]
Frequency = Union[BaseOffset, Timedelta]
FrequencyLike = Union[BaseOffset, TimedeltaLike]
TimezoneLike = Union[None, str, int, float, timedelta, tzinfo]
TimeLike = Union[str, time]
PandasFrequency = Union[BaseOffset, PandasTimedelta]
PandasDatetimeIndex = Union[DatetimeIndex, PeriodIndex]
AnyPandasFrequency = Union[None, int, float, PandasFrequency]
DTCLike = Union[None, str, int, time, date, Datetime, DTC, DTCNT]


class SupportsTZInfoT(Protocol):
    @property
    def tzinfo(self) -> tzinfo: ...


# Indexing
Slice = Union[slice, hslice]
PandasIndexingFunc = Callable[[SeriesFrame], MaybeSeriesFrame]

# Grouping
PandasGroupByLike = Union[PandasGroupBy, PandasResampler, FrequencyLike]
GroupByLike = Union[None, bool, MaybeLevelSequence, IndexLike, CustomTemplate]
AnyGroupByLike = Union[Grouper, PandasGroupByLike, GroupByLike]
GroupBy = Union[None, bool, Index]
AnyRuleLike = Union[Resampler, PandasResampler, FrequencyLike, IndexLike]
GroupIdxs = Array1d
GroupLens = Array1d
GroupMap = Tuple[GroupIdxs, GroupLens]

# Wrapping
NameIndex = Union[None, Any, Index]

# Search
PathKeyToken = Hashable
PathKeyTokens = Sequence[Hashable]
PathKey = Tuple[PathKeyToken, ...]
MaybePathKey = Union[None, PathKeyToken, PathKey]
PathLikeKey = Union[MaybePathKey, Path]
PathLikeKeys = Sequence[PathLikeKey]
PathMoveDict = Dict[PathLikeKey, PathLikeKey]
PathRenameDict = Dict[PathLikeKey, PathKeyToken]
PathDict = Dict[PathLikeKey, Any]

# Config
DictLike = Union[None, dict]
DictLikeSequence = MaybeSequence[DictLike]
Args = Tuple[Any, ...]
ArgsLike = Union[None, Args]
Kwargs = Dict[str, Any]
KwargsLike = Union[None, Kwargs]
KwargsLikeSequence = MaybeSequence[KwargsLike]
ArgsKwargs = Tuple[Args, Kwargs]
PathLike = Union[str, Path]
_SettingsPath = Union[None, MaybeList[PathLikeKey], Dict[Hashable, PathLikeKey]]
SettingsPath = ClassVar[_SettingsPath]
ExtSettingsPaths = List[Tuple[type, _SettingsPath]]
SpecSettingsPaths = Dict[PathLikeKey, MaybeList[PathLikeKey]]
WriteableAttrs = ClassVar[Optional[Set[str]]]
ExpectedKeysMode = ClassVar[str]
ExpectedKeys = ClassVar[Optional[Set[str]]]

# Data
Column = Key = Feature = Symbol = Hashable
Columns = Keys = Features = Symbols = Sequence[Hashable]
MaybeColumns = MaybeKeys = MaybeFeatures = MaybeSymbols = Union[Hashable, Sequence[Hashable]]
KeyData = FeatureData = SymbolData = Union[None, SeriesFrame, Tuple[SeriesFrame, Kwargs]]
PullOutput = Union[Data, List[Any]]

# Plotting
TraceName = Union[str, None]
TraceNames = MaybeSequence[TraceName]

# Combining
R0ApplyFunc = Callable[[int, VarArg()], None]
R1ApplyFunc = Callable[[int, VarArg()], Array]
RMApplyFunc = Callable[[int, VarArg()], Tuple[Array, ...]]
CApplyFunc = Union[R0ApplyFunc, R1ApplyFunc, RMApplyFunc]
CombineFunc = Callable[[Any, Any, VarArg()], Any]
PyCombineFunc = Callable[[Any, Any, VarArg(), KwArg()], Any]
AnyCombineFunc = Union[CombineFunc, PyCombineFunc]

# Generic
MapFunc = Callable[[Scalar, VarArg()], Scalar]
MapMetaFunc = Callable[[int, int, Scalar, VarArg()], Scalar]
ApplyFunc = Callable[[Array1d, VarArg()], MaybeArray]
ApplyMetaFunc = Callable[[int, VarArg()], MaybeArray]
ReduceFunc = Callable[[Array1d, VarArg()], Scalar]
ReduceMetaFunc = Callable[[int, VarArg()], Scalar]
ReduceToArrayFunc = Callable[[Array1d, VarArg()], Array1d]
ReduceToArrayMetaFunc = Callable[[int, VarArg()], Array1d]
ReduceGroupedFunc = Callable[[Array2d, VarArg()], Scalar]
ReduceGroupedMetaFunc = Callable[[GroupIdxs, int, VarArg()], Scalar]
ReduceGroupedToArrayFunc = Callable[[Array2d, VarArg()], Array1d]
ReduceGroupedToArrayMetaFunc = Callable[[GroupIdxs, int, VarArg()], Array1d]
RangeReduceMetaFunc = Callable[[int, int, int, VarArg()], Scalar]
ProximityReduceMetaFunc = Callable[[int, int, int, int, VarArg()], Scalar]
GroupByReduceMetaFunc = Callable[[GroupIdxs, int, int, VarArg()], Scalar]
GroupSqueezeMetaFunc = Callable[[int, GroupIdxs, int, VarArg()], Scalar]
GroupByTransformFunc = Callable[[Array2d, VarArg()], MaybeArray]
GroupByTransformMetaFunc = Callable[[GroupIdxs, int, VarArg()], MaybeArray]

AnyMapFunc = Union[MapFunc, MapMetaFunc]
AnyApplyFunc = Union[ApplyFunc, ApplyMetaFunc]
AnyReduceFunc = Union[ReduceFunc, ReduceMetaFunc]
AnyGroupByReduceFunc = Union[ReduceFunc, GroupByReduceMetaFunc]
AnyGroupByTransformFunc = Union[GroupByTransformFunc, GroupByTransformMetaFunc]
AnyResampleReduceFunc = Union[ReduceFunc, GroupByReduceMetaFunc, RangeReduceMetaFunc]
AnyFlexReduceFunc = Union[
    ReduceFunc,
    ReduceMetaFunc,
    ReduceToArrayFunc,
    ReduceToArrayMetaFunc,
    ReduceGroupedFunc,
    ReduceGroupedMetaFunc,
    ReduceGroupedToArrayFunc,
    ReduceGroupedToArrayMetaFunc,
]
AnyProximityReduceFunc = Union[ReduceFunc, ProximityReduceMetaFunc]
AnyGroupSqueezeFunc = Union[ReduceFunc, GroupSqueezeMetaFunc]
AnyRangeReduceFunc = Union[ReduceFunc, RangeReduceMetaFunc]


class TransformerT(Protocol):
    def __init__(self, **kwargs) -> None: ...
    def transform(self, *args, **kwargs) -> Array2d: ...
    def fit_transform(self, *args, **kwargs) -> Array2d: ...


# Signals
PlaceFunc = Callable[[NamedTuple, VarArg()], int]
RankFunc = Callable[[NamedTuple, VarArg()], int]

# Records
RecordsMapFunc = Callable[[np.void, VarArg()], Scalar]
RecordsMapMetaFunc = Callable[[int, VarArg()], Scalar]
MappedReduceMetaFunc = Callable[[GroupIdxs, int, VarArg()], Scalar]
MappedReduceToArrayMetaFunc = Callable[[GroupIdxs, int, VarArg()], Array1d]

AnyRecordsMapFunc = Union[RecordsMapFunc, RecordsMapMetaFunc]
AnyMappedReduceFunc = Union[
    ReduceFunc, MappedReduceMetaFunc, ReduceToArrayFunc, MappedReduceToArrayMetaFunc
]

# Indicators
ParamValue = Any
ParamValues = Sequence[ParamValue]
MaybeParamValues = MaybeSequence[ParamValue]
MaybeParams = Sequence[MaybeParamValues]
Params = Sequence[ParamValues]
ParamsOrLens = Sequence[Union[ParamValues, int]]
ParamsOrDict = Union[Params, Dict[Hashable, ParamValues]]
ParamGrid = Union[ParamsOrLens, Dict[Hashable, ParamsOrLens]]
ParamComb = Sequence[ParamValue]
ParamCombOrDict = Union[ParamComb, Dict[Hashable, ParamValue]]
IFCacheOutput = Any
IFRawOutput = Tuple[
    List[Array2d],
    List[Tuple[ParamValue, ...]],
    int,
    List[Any],
]
IFArrayList = List[Array2d]
IFInputMapper = Optional[Array1d]
IFParamList = List[List[ParamValue]]
IFMapperList = List[Index]
IFOtherList = List[Any]
IFPipelineOutput = Tuple[
    ArrayWrapper,
    IFArrayList,
    IFInputMapper,
    IFArrayList,
    IFArrayList,
    IFParamList,
    IFMapperList,
    IFOtherList,
]
IFRunOutput = Union[IndicatorBase, Tuple[Any, ...], IFRawOutput, IFCacheOutput]
IFRunCombsOutput = Tuple[IndicatorBase, ...]

# Mappings
MappingLike = Union[str, Mapping, NamedTuple, EnumMeta, IndexLike]
RecordsLike = Union[SeriesFrame, RecordArray, Sequence[MappingLike]]

# Annotations
Annotation = object
Annotations = Dict[str, Annotation]

# Parsing
AnnArgs = Dict[str, Kwargs]
FlatAnnArgs = Dict[str, Kwargs]
AnnArgQuery = Union[int, str, Regex]

# Execution
FuncArgs = Tuple[Callable, Args, Kwargs]
FuncsArgs = Iterable[FuncArgs]
TaskLike = Union[FuncArgs, Task]
TasksLike = Iterable[TaskLike]
ExecutionEngineLike = Union[str, type, ExecutionEngine, Callable]
ExecResult = Any
ExecResults = List[Any]

# JIT
JittedOption = Union[None, bool, str, Callable, Kwargs]
JitterLike = Union[str, Jitter, Type[Jitter]]
TaskId = Union[Hashable, Callable]

# Merging
MergeFuncLike = MaybeSequence[Union[None, str, Callable, MergeFunc]]
MergeResult = Any
MergeableResults = Union[ExecResults, MergeResult]

# Chunking
SizeFunc = Callable[[AnnArgs], int]
SizeLike = Union[int, str, Sizer, SizeFunc]
ChunkMetaFunc = Callable[[AnnArgs], Iterable[ChunkMeta]]
ChunkMetaLike = Union[Iterable[ChunkMeta], ChunkMetaGenerator, ChunkMetaFunc]
TakeSpec = Union[None, Type[NotChunked], Type[ChunkTaker], NotChunked, ChunkTaker]
ArgTakeSpec = Mapping[AnnArgQuery, TakeSpec]
ArgTakeSpecFunc = Callable[[AnnArgs, ChunkMeta], Tuple[Args, Kwargs]]
ArgTakeSpecLike = Union[Sequence[TakeSpec], ArgTakeSpec, ArgTakeSpecFunc, CustomTemplate]
MappingTakeSpec = Mapping[Hashable, TakeSpec]
SequenceTakeSpec = Sequence[TakeSpec]
ContainerTakeSpec = Union[MappingTakeSpec, SequenceTakeSpec]
ChunkedOption = Union[None, bool, str, Callable, Kwargs]

# Decorators
ClassWrapper = Callable[[Type[T]], Type[T]]
FlexClassWrapper = Union[Callable[[Type[T]], Type[T]], Type[T]]
UnaryTranslateFunc = Callable[[Any, Callable], Any]
BinaryTranslateFunc = Callable[[Any, Any, Callable], Any]

# Splitting
FixRangeLike = Union[Slice, Sequence[int], Sequence[bool], Callable, CustomTemplate, FixRange]
RelRangeLike = Union[int, float, Callable, CustomTemplate, RelRange]
RangeLike = Union[FixRangeLike, RelRangeLike]
ReadyRangeLike = Union[slice, Array1d]
FixSplit = Sequence[FixRangeLike]
SplitLike = Union[str, int, float, MaybeSequence[RangeLike]]
Splits = Sequence[SplitLike]
SplitsArray = Array2d
SplitsMask = Array3d
BoundsArray = Array3d

# Staticization
StaticizedOption = Union[None, bool, Kwargs, TaskId]

# Selection
Selection = Union[PosSel, LabelSel, MaybeIterable[Union[PosSel, LabelSel, Hashable]]]

# Knowledge
AssetFuncLike = Union[str, Type[AssetFunc], FuncArgs, Task, Callable]
MaybeAsset = Union[None, T, dict, list, Iterator[T]]
MaybeKnowledgeAsset = MaybeAsset[KnowledgeAsset]
MaybeVBTAsset = MaybeAsset[VBTAsset]
MaybePagesAsset = MaybeAsset[PagesAsset]
MaybeMessagesAsset = MaybeAsset[MessagesAsset]
MaybeExamplesAsset = MaybeAsset[ExamplesAsset]
ContentFormatterLike = Union[None, str, MaybeType[ContentFormatter]]
TokenizerLike = Union[None, str, MaybeType[Tokenizer]]
Token = int
Tokens = List[Token]
EmbeddingsLike = Union[None, str, MaybeType[Embeddings]]
CompletionsLike = Union[None, str, MaybeType[Completions]]
ChatMessage = dict
ChatMessages = List[ChatMessage]
ChatHistory = MutableSequence[ChatMessage]
ChatOutput = Union[Optional[Path], Tuple[Optional[Path], Any]]
MaybeChatOutput = Union[ChatOutput, Tuple[ChatOutput, Completions]]
TextSplitterLike = Union[None, str, MaybeType[TextSplitter]]
TSSpan = Tuple[int, int]
TSSpanChunks = Iterator[TSSpan]
TSSegment = Tuple[int, int, bool]
TSSegmentChunks = Iterator[TSSpan]
TSTextChunks = Iterator[str]
TSSourceChunk = Tuple[str, int]
TSSourceChunks = Iterator[TSSourceChunk]
ShouldSplitCodeFunc = Callable[[ast.AST, int, int, int], bool]
ShouldSplitCode = Union[bool, int, ShouldSplitCodeFunc]
ShouldSplitMarkdownFunc = Callable[[str, int, int, int], bool]
ShouldSplitMarkdown = Union[bool, int, ShouldSplitMarkdownFunc]
ObjectStoreLike = Union[None, str, MaybeType[ObjectStore]]
SplitDocuments = List[List[StoreDocument]]
EmbeddedDocuments = List[EmbeddedDocument]
ScoredDocuments = List[Union[float, ScoredDocument]]
RankedDocuments = List[Union[StoreDocument, ScoredDocument]]
TopKLike = Union[None, int, float, str, Callable]

# Chaining
PipeFunc = Union[str, Callable, Tuple[Union[str, Callable], str]]
PipeTask = Union[PipeFunc, Tuple[PipeFunc, Args, Kwargs], Task]
PipeTasks = Iterable[PipeTask]

# Pickling
CompressionLike = Union[None, bool, str]

# Source
RefactorSourceOutput = Union[None, str, Path, Tuple[str, Path], Tuple[Path, Path]]
RefactorSourceOutputs = List[Tuple[Any, RefactorSourceOutput]]
MaybeRefactorSourceOutput = Union[RefactorSourceOutput, RefactorSourceOutputs]

# Simulation
SignalFunc = Callable[[SignalContext, VarArg()], Tuple[bool, bool, bool, bool]]
PostSignalFunc = Callable[[PostSignalContext, VarArg()], None]
PostSignalSegmentFunc = Callable[[SignalSegmentContext, VarArg()], None]
AdjustFunc = Callable[[SignalContext, VarArg()], None]
PreSimFunc = Callable[[SimulationContext, VarArg()], Args]
PostSimFunc = Callable[[SimulationContext, VarArg()], None]
PreGroupFunc = Callable[[GroupContext, VarArg()], Args]
PostGroupFunc = Callable[[GroupContext, VarArg()], None]
PreRowFunc = Callable[[RowContext, VarArg()], Args]
PostRowFunc = Callable[[RowContext, VarArg()], None]
PreSegmentFunc = Callable[[SegmentContext, VarArg()], Args]
PostSegmentFunc = Callable[[SegmentContext, VarArg()], None]
OrderFunc = Callable[[OrderContext, VarArg()], Order]
FlexOrderFunc = Callable[[FlexOrderContext, VarArg()], Tuple[int, Order]]
PostOrderFunc = Callable[[PostOrderContext, VarArg()], None]

# Portfolio optimization
AllocateFunc = Callable[[int, int, VarArg()], MaybeArray]
OptimizeFunc = Callable[[int, int, int, VarArg()], MaybeArray]
