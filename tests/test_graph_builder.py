import pytest

from src.graph.graph_builder import GraphBuilder, build_cad_graph


class DummyFeatureVector:
    def __init__(self, x, y=None):
        self.x = x
        self.y = y


class DummyFace:
    def __init__(
        self,
        face_id,
        feature_vector,
        neighbors=None,
        surface_type="PLANE",
        area=1.0,
        perimeter=4.0,
        compactness=0.8,
        complexity_score=0.2,
        density_label="LOW",
        density_id=0,
    ):
        self.face_id = face_id
        self.feature_vector = feature_vector
        self.neighbors = neighbors or []
        self.surface_type = surface_type
        self.area = area
        self.perimeter = perimeter
        self.compactness = compactness
        self.complexity_score = complexity_score
        self.density_label = density_label
        self.density_id = density_id


class DummyModel:
    def __init__(self, faces, metadata=None):
        self.faces = faces
        self.metadata = metadata or {}


def test_build_graph_creates_nodes():
    model = DummyModel(
        faces=[
            DummyFace(10, DummyFeatureVector([1.0, 2.0], y=0)),
            DummyFace(20, DummyFeatureVector([3.0, 4.0], y=1)),
        ],
        metadata={"model_name": "test_model"},
    )

    graph = GraphBuilder().build_graph(model)

    assert graph.graph_id == "test_model"
    assert graph.node_count() == 2
    assert graph.edge_count() == 0

    assert graph.nodes[0].node_id == 0
    assert graph.nodes[0].face_id == 10
    assert graph.nodes[0].features == [1.0, 2.0]
    assert graph.nodes[0].label == 0

    assert graph.nodes[1].node_id == 1
    assert graph.nodes[1].face_id == 20
    assert graph.nodes[1].features == [3.0, 4.0]
    assert graph.nodes[1].label == 1


def test_build_graph_creates_bidirectional_edges():
    model = DummyModel(
        faces=[
            DummyFace(0, DummyFeatureVector([1.0], y=0), neighbors=[1]),
            DummyFace(1, DummyFeatureVector([2.0], y=1), neighbors=[0]),
        ]
    )

    graph = GraphBuilder(bidirectional=True).build_graph(model)

    assert graph.edge_count() == 2
    assert sorted(graph.edge_index()) == [(0, 1), (1, 0)]


def test_build_graph_creates_unidirectional_edges():
    model = DummyModel(
        faces=[
            DummyFace(0, DummyFeatureVector([1.0], y=0), neighbors=[1]),
            DummyFace(1, DummyFeatureVector([2.0], y=1), neighbors=[]),
        ]
    )

    graph = GraphBuilder(bidirectional=False).build_graph(model)

    assert graph.edge_count() == 1
    assert graph.edge_index() == [(0, 1)]


def test_build_graph_removes_duplicate_edges():
    model = DummyModel(
        faces=[
            DummyFace(0, DummyFeatureVector([1.0], y=0), neighbors=[1, 1]),
            DummyFace(1, DummyFeatureVector([2.0], y=1), neighbors=[0]),
        ]
    )

    graph = GraphBuilder(bidirectional=True).build_graph(model)

    assert graph.edge_count() == 2
    assert sorted(graph.edge_index()) == [(0, 1), (1, 0)]


def test_build_graph_ignores_unknown_neighbors():
    model = DummyModel(
        faces=[
            DummyFace(0, DummyFeatureVector([1.0], y=0), neighbors=[99]),
            DummyFace(1, DummyFeatureVector([2.0], y=1), neighbors=[]),
        ]
    )

    graph = GraphBuilder().build_graph(model)

    assert graph.node_count() == 2
    assert graph.edge_count() == 0


def test_build_graph_ignores_self_loops():
    model = DummyModel(
        faces=[
            DummyFace(0, DummyFeatureVector([1.0], y=0), neighbors=[0]),
        ]
    )

    graph = GraphBuilder().build_graph(model)

    assert graph.node_count() == 1
    assert graph.edge_count() == 0


def test_build_graph_face_metadata_is_preserved():
    face = DummyFace(
        face_id=0,
        feature_vector=DummyFeatureVector([1.0], y=2),
        neighbors=[],
        surface_type="CYLINDER",
        area=25.0,
        perimeter=10.0,
        compactness=0.6,
        complexity_score=0.55,
        density_label="HIGH",
        density_id=2,
    )

    graph = GraphBuilder().build_graph(DummyModel([face]))

    metadata = graph.nodes[0].metadata

    assert metadata["surface_type"] == "CYLINDER"
    assert metadata["area"] == 25.0
    assert metadata["perimeter"] == 10.0
    assert metadata["compactness"] == 0.6
    assert metadata["complexity_score"] == 0.55
    assert metadata["density_label"] == "HIGH"
    assert metadata["density_id"] == 2


def test_build_graph_edge_metadata_is_preserved():
    model = DummyModel(
        faces=[
            DummyFace(10, DummyFeatureVector([1.0], y=0), neighbors=[20]),
            DummyFace(20, DummyFeatureVector([2.0], y=1), neighbors=[]),
        ]
    )

    graph = GraphBuilder(bidirectional=False).build_graph(model)

    edge = graph.edges[0]

    assert edge.metadata["source_face_id"] == 10
    assert edge.metadata["target_face_id"] == 20
    assert edge.metadata["edge_type"] == "FACE_ADJACENCY"
    assert edge.features == [1.0]


def test_build_graph_rejects_none_model():
    with pytest.raises(ValueError, match="CADModel cannot be None"):
        GraphBuilder().build_graph(None)


def test_build_graph_requires_faces_attribute():
    class DummyModelWithoutFaces:
        pass

    with pytest.raises(AttributeError, match="faces"):
        GraphBuilder().build_graph(DummyModelWithoutFaces())


def test_build_graph_rejects_empty_faces():
    with pytest.raises(ValueError, match="at least one face"):
        GraphBuilder().build_graph(DummyModel([]))


def test_build_graph_requires_feature_vector():
    class FaceWithoutFeatureVector:
        def __init__(self):
            self.face_id = 0
            self.neighbors = []

    model = DummyModel([FaceWithoutFeatureVector()])

    with pytest.raises(AttributeError, match="feature_vector"):
        GraphBuilder().build_graph(model)


def test_build_graph_requires_feature_vector_x():
    class BadFeatureVector:
        pass

    face = DummyFace(0, BadFeatureVector(), neighbors=[])

    model = DummyModel([face])

    with pytest.raises(AttributeError, match="feature_vector must contain 'x'"):
        GraphBuilder().build_graph(model)


def test_build_cad_graph_convenience_function():
    model = DummyModel(
        faces=[
            DummyFace(0, DummyFeatureVector([1.0], y=0), neighbors=[]),
        ],
        metadata={"model_name": "convenience_model"},
    )

    graph = build_cad_graph(model)

    assert graph.graph_id == "convenience_model"
    assert graph.node_count() == 1
    assert graph.nodes[0].features == [1.0]


def test_graph_id_fallback_to_cad_graph():
    model = DummyModel(
        faces=[
            DummyFace(0, DummyFeatureVector([1.0], y=0), neighbors=[]),
        ]
    )

    graph = GraphBuilder().build_graph(model)

    assert graph.graph_id == "cad_graph"


def test_graph_id_from_model_attribute():
    class ModelWithName:
        def __init__(self):
            self.faces = [
                DummyFace(0, DummyFeatureVector([1.0], y=0), neighbors=[]),
            ]
            self.model_name = "attribute_model"

    graph = GraphBuilder().build_graph(ModelWithName())

    assert graph.graph_id == "attribute_model"