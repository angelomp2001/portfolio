import os
import ast
from pathlib import Path

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D


# ---------------------------
# CONFIGURATION
# ---------------------------

PROJECT_ROOT: str | None = r"C:\Users\Angelo\Documents\github\portfolio\portfolio\project_3"

# If None -> show interactively; else save to this file (e.g. "module_deps.png")
OUTPUT_IMAGE: str | None = "dag.png"

# File System Parameters
SKIP_DIRS: set[str] = {".git", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache"}

# Plot Metadata
PLOT_TITLE: str = "Module Dependencies (A → B means A imports B)\nEach module contains child boxes for its top-level classes, functions, and dependent variables"

# Visualization parameters
FIGURE_SIZE: tuple[int, int] = (24, 16)

# Layout parameters
COL_STEP: float = 15.0 # horizontal distance between columns
GROUP_GAP: float = 2.0 # vertical gap between module containers
CHILD_BOX_HEIGHT: float = 0.5 # height of child boxes
CHILD_VERTICAL_GAP: float = 0.15 # vertical gap between child boxes
TITLE_HEIGHT: float = 0.7 # height of title
SIDE_PADDING: float = 0.4 # padding around child boxes
CONTAINER_WIDTH: float = 6.0 # width of module containers
DESIRED_MIN_CENTER: float = 3.0 # minimum x-coordinate for the leftmost module container
HORIZ_PAD: float = 1.5 # horizontal padding for the entire plot axes
VERT_PAD: float = 2.0 # vertical padding for the entire plot axes

# Styling: Module Container
CONTAINER_LINEWIDTH: float = 1.2 # border thickness of module containers
CONTAINER_EDGECOLOR: str = "black" # border color of module containers
CONTAINER_FACECOLOR: str = "lightblue" # background color of module containers

# Styling: Module Title
TITLE_FONTSIZE: int = 9 # font size of module titles
TITLE_FONTWEIGHT: str = "bold" # font weight of module titles

# Styling: Child Node
CHILD_LINEWIDTH: float = 1.0 # border thickness of child nodes
CHILD_EDGECOLOR: str = "black" # border color of child nodes
CHILD_FACECOLOR: str = "white" # background color of child nodes
CHILD_FONTSIZE: int = 7 # font size of child nodes

# Styling: Edges
EDGE_CONNECTIONSTYLE: str = "arc3,rad=0.1" # connection style of edges
EDGE_ARROWSTYLE: str = "-|>" # arrow style of edges

# Styling: Child-to-Child Edge
CHILD_EDGE_COLOR: str = "blue" # color of child-to-child edges
CHILD_EDGE_ALPHA: float = 0.5 # alpha of child-to-child edges
CHILD_EDGE_MUTATION_SCALE: int = 12 # arrow head size of child-to-child edges

# Styling: Module-to-Module Edge
MODULE_EDGE_COLOR: str = "black" # color of module-to-module edges
MODULE_EDGE_ALPHA: float = 0.3 # alpha of module-to-module edges
MODULE_EDGE_MUTATION_SCALE: int = 15 # arrow head size of module-to-module edges


# ---------------------------
# UTILITIES
# ---------------------------

def find_python_files(root: Path):
    """Yield all .py files under root (excluding common virtualenv/hidden dirs)."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for file in filenames:
            if file.endswith(".py") and not file.startswith("."):
                yield Path(dirpath) / file


def module_name_from_path(root: Path, file_path: Path) -> str:
    """Create dotted module name from a file path relative to root."""
    rel = file_path.relative_to(root).with_suffix("")
    return ".".join(rel.parts)


# ---------------------------
# AST ANALYSIS – modules & imports
# ---------------------------

class DependencyScanner(ast.NodeVisitor):
    """Collect imports for a single module (file)."""

    def __init__(self, module_name: str, project_modules: set[str], graph: nx.DiGraph):
        self.module_name = module_name
        self.project_modules = project_modules  # dotted names of all project modules
        self.graph = graph

        # Ensure module node exists
        if not self.graph.has_node(self.module_name):
            self.graph.add_node(self.module_name, kind="module")

    def _add_import_edge(self, imported: str):
        """
        Add module -> module edge if imported name corresponds to a project module.
        We only care about project modules, not external libraries.
        """
        if imported in self.project_modules:
            if not self.graph.has_node(imported):
                self.graph.add_node(imported, kind="module")
            self.graph.add_edge(self.module_name, imported, kind="import")

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            # import pkg.mod as m  -> imported "pkg.mod"
            imported = alias.name
            self._add_import_edge(imported)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        # from pkg import mod       -> node.module = "pkg"
        # from pkg.mod import foo   -> node.module = "pkg.mod"
        if node.module is None:
            # from . import something – ignore for now
            return

        base = node.module  # may be "pkg", "pkg.mod", etc.
        # Try using the full base as a module
        self._add_import_edge(base)

        # Also try base + imported names, in case they refer to project modules
        for alias in node.names:
            candidate = f"{base}.{alias.name}"
            self._add_import_edge(candidate)

        self.generic_visit(node)


# ---------------------------
# AST ANALYSIS – public API per module
# ---------------------------

class ModuleContentScanner(ast.NodeVisitor):
    """
    Collect top-level classes, functions, and explicit imports for a module.
    Flags ABCs for Interface Independence analysis.
    """

    def __init__(self, project_modules: set[str]):
        self.project_modules = project_modules
        self.classes: list[str] = []
        self.functions: list[str] = []
        self.variables: list[str] = []
        self.imports: list[str] = []

    def visit_ClassDef(self, node: ast.ClassDef):
        is_abstract = any(
            (isinstance(b, ast.Name) and b.id == "ABC") or
            (isinstance(b, ast.Attribute) and b.attr == "ABC")
            for b in node.bases
        )
        
        name = f"<<Interface>> {node.name}" if is_abstract else node.name
        self.classes.append(name)
        # Don't descend into methods for this simple view

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.functions.append(node.name)

    def visit_Assign(self, node: ast.Assign):
        self.variables.extend(t.id for t in node.targets if isinstance(t, ast.Name))

    def visit_AnnAssign(self, node: ast.AnnAssign):
        if isinstance(node.target, ast.Name):
            self.variables.append(node.target.id)

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            if alias.name in self.project_modules:
                name = alias.asname or alias.name
                self.imports.append(name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module in self.project_modules:
            for alias in node.names:
                name = alias.asname or alias.name
                self.imports.append(name)
        self.generic_visit(node)


def build_module_graph(project_root: Path):
    """
    Returns:
      - graph: DiGraph with module->module import edges
      - module_classes: dict[module_name] = [ClassName, ...]
      - module_functions: dict[module_name] = [func_name, ...]
      - module_variables: dict[module_name] = [var_name, ...]
      - module_imports: dict[module_name] = [func_name, ...]
    """
    # === STEP 1: Discover Python Files ===
    # We find all .py files in the project directory, skipping specified ignored directories (like .venv).
    # Then we map each file path to a usable "module name" (e.g., config, utils.helpers).
    py_files = list(find_python_files(project_root))
    project_modules: dict[Path, str] = {
        f: module_name_from_path(project_root, f) for f in py_files
    }

    print("Discovered modules (path -> module name):")
    for path, mod_name in project_modules.items():
        print(f"  {path} -> {mod_name}")

    graph = nx.DiGraph()

    # Add ALL modules as nodes up front
    for mod_name in project_modules.values():
        graph.add_node(mod_name, kind="module")

    module_classes: dict[str, list[str]] = {}
    module_functions: dict[str, list[str]] = {}
    module_variables: dict[str, list[str]] = {}
    module_imports: dict[str, list[str]] = {}

    for file_path, module_name in project_modules.items():
        with file_path.open("r", encoding="utf-8") as f:
            try:
                source = f.read()
                tree = ast.parse(source, filename=str(file_path))
            except SyntaxError as e:
                print(f"Skipping {file_path} due to SyntaxError: {e}")
                continue

        # === STEP 2: Extract Dependencies (Imports) ===
        # For each discovered file, we use the DependencyScanner to read the file's code (using the ast library).
        # It looks specifically for `import xyz` and `from xyz import abc` statements to build a graph of dependencies.
        import_analyzer = DependencyScanner(
            module_name,
            set(project_modules.values()),
            graph,
        )
        import_analyzer.visit(tree)

        # === STEP 3: Extract Module Contents ===
        # Using the ModuleContentScanner, we read each file to identify the classes, functions,
        # and variables defined inside it so that we can visually list them inside each module box.
        api_analyzer = ModuleContentScanner(set(project_modules.values()))
        api_analyzer.visit(tree)
        module_classes[module_name] = api_analyzer.classes
        module_functions[module_name] = api_analyzer.functions
        module_variables[module_name] = api_analyzer.variables
        module_imports[module_name] = api_analyzer.imports

    # === STEP 4: Filter and Refine Module Children ===
    # We cross-reference all exported elements to ensure we only display variables if they are actually used
    # by other modules as an injected dependency.
    all_imported_names = {i for imports in module_imports.values() for i in imports}
        
    for mod_name in module_variables:
        module_variables[mod_name] = [
            v for v in module_variables[mod_name] 
            if v in all_imported_names
        ]

    # We also format functions by appending "()" so they are visually distinguishable from variables.
    all_function_names = {f for funcs in module_functions.values() for f in funcs}
        
    for mod_name in module_functions:
        module_functions[mod_name] = [f"{f}()" for f in module_functions[mod_name]]
        
    for mod_name in module_imports:
        module_imports[mod_name] = [
            f"{i}()" if i in all_function_names else i 
            for i in module_imports[mod_name]
        ]

    return graph, module_classes, module_functions, module_variables, module_imports


# ---------------------------
# VISUALIZATION – modules only
# ---------------------------

def visualize_module_dependencies(
    graph: nx.DiGraph,
    module_classes: dict[str, list[str]],
    module_functions: dict[str, list[str]],
    module_variables: dict[str, list[str]],
    module_imports: dict[str, list[str]],
    output: Path | None = None,
):
    """
    Left-to-right module dependency graph.

    - Outer node: module (single light-blue container box).
    - Edge A -> B: module A imports module B.
    - Inside each module, each top-level class/function is drawn
      as its own white child box.
    """
    
    # === STEP 5: Visualizing the Data ===
    # Using matplotlib and networkx, we lay out the graph dependencies and draw the containers and arrows.
    plt.close("all")
    plt.figure(figsize=FIGURE_SIZE)
    ax = plt.gca()

    # ---- 1. Prepare module dependency graph ----
    modules = [n for n, d in graph.nodes(data=True) if d.get("kind") == "module"]

    dep_edges = [
        (u, v)
        for u, v, d in graph.edges(data=True)
        if d.get("kind") == "import" and u in modules and v in modules
    ]
    dep_graph = nx.DiGraph()
    dep_graph.add_nodes_from(modules)
    dep_graph.add_edges_from(dep_edges)

    try:
        topo_order = list(nx.topological_sort(dep_graph))
    except nx.NetworkXUnfeasible:
        topo_order = modules

    layer = {n: 0 for n in modules}
    for n in topo_order:
        preds = list(dep_graph.predecessors(n))
        if preds:
            layer[n] = max(layer[p] + 1 for p in preds)

    layers: dict[int, list[str]] = {}
    for n, lvl in layer.items():
        layers.setdefault(lvl, []).append(n)

    # ---- 2. Layout parameters ----
    col_step = COL_STEP            # more horizontal distance between columns
    group_gap = GROUP_GAP           # vertical gap between module containers
    child_box_height = CHILD_BOX_HEIGHT
    child_vertical_gap = CHILD_VERTICAL_GAP
    title_height = TITLE_HEIGHT
    side_padding = SIDE_PADDING
    container_width = CONTAINER_WIDTH

    def get_module_children(m: str) -> list[str]:
        return module_classes.get(m, []) + module_functions.get(m, []) + module_variables.get(m, []) + module_imports.get(m, [])

    # Precompute exact container heights per module
    container_height: dict[str, float] = {}
    for m in modules:
        children = get_module_children(m)
        n_children = len(children)
        if n_children == 0:
            h = title_height + 0.8
        else:
            children_height = (
                n_children * child_box_height
                + max(0, n_children - 1) * child_vertical_gap
            )
            h = title_height + children_height + 0.8
        container_height[m] = h

    # ---- 3. Compute module centers per layer ----
    mod_center: dict[str, tuple[float, float]] = {}

    # First compute provisional x positions (without shift)
    provisional_center: dict[str, tuple[float, float]] = {}
    for col, lvl in enumerate(sorted(layers.keys())):
        mods = sorted(layers[lvl])

        total_height = sum(container_height[m] for m in mods) + group_gap * (len(mods) - 1)
        y_top = total_height / 2.0

        current_y_top = y_top
        for m in mods:
            h = container_height[m]
            cy = current_y_top - h / 2.0
            cx = col * col_step
            provisional_center[m] = (cx, cy)
            current_y_top -= h + group_gap

    # Decide global horizontal shift so leftmost box is safely inside frame
    if provisional_center:
        min_x = min(cx for cx, _ in provisional_center.values())
        # we want the leftmost module center to be at x >= 3.0
        desired_min_center = DESIRED_MIN_CENTER
        shift_x = max(0.0, desired_min_center - min_x)
    else:
        shift_x = 0.0

    # Apply shift and store final centers
    for m, (cx, cy) in provisional_center.items():
        mod_center[m] = (cx + shift_x, cy)

    # ---- 4. Draw containers and child boxes ----
    child_box_coords: dict[str, dict[str, tuple[float, float, float]]] = {}
    for m, (cx, cy) in mod_center.items():
        child_box_coords[m] = {}
        mod_label = f"{m.split('.')[-1]}.py"
        children = get_module_children(m)
        h = container_height[m]

        bottom = cy - h / 2.0

        # Outer container
        container = plt.Rectangle(
            (cx - container_width / 2.0, bottom),
            container_width,
            h,
            linewidth=CONTAINER_LINEWIDTH,
            edgecolor=CONTAINER_EDGECOLOR,
            facecolor=CONTAINER_FACECOLOR,
            zorder=1,
        )
        ax.add_patch(container)

        # Title
        title_y = cy + h / 2.0 - 0.3
        ax.text(
            cx,
            title_y,
            mod_label,
            fontsize=TITLE_FONTSIZE,
            ha="center",
            va="top",
            fontweight=TITLE_FONTWEIGHT,
            zorder=2,
        )

        # Children stacked inside
        current_y = title_y - title_height
        child_width = container_width - 2 * side_padding

        for name in children:
            box_center_y = current_y - child_box_height / 2.0

            x_left = cx - child_width / 2.0
            x_right = cx + child_width / 2.0
            child_box_coords[m][name] = (x_left, x_right, box_center_y)

            child = plt.Rectangle(
                (x_left, box_center_y - child_box_height / 2.0),
                child_width,
                child_box_height,
                linewidth=CHILD_LINEWIDTH,
                edgecolor=CHILD_EDGECOLOR,
                facecolor=CHILD_FACECOLOR,
                zorder=1.5,
            )
            ax.add_patch(child)

            ax.text(
                x_left + 0.1,
                box_center_y,
                name,
                fontsize=CHILD_FONTSIZE,
                ha="left",
                va="center",
                zorder=2,
            )

            current_y -= child_box_height + child_vertical_gap

    # ---- 5. Draw edges ----
    def strip_prefix(name: str) -> str:
        if name.startswith("<<Interface>> "): return name.replace("<<Interface>> ", "")
        return name

    exports_map = {}
    for m in modules:
        for exp in module_classes.get(m, []) + module_functions.get(m, []) + module_variables.get(m, []):
            exports_map.setdefault(strip_prefix(exp), []).append((m, exp))

    for u, v in dep_edges:
        cx_u, cy_u = mod_center[u]
        h_u = container_height[u]
        
        cx_v, cy_v = mod_center[v]
        h_v = container_height[v]

        child_edge_drawn = False
        
        if u in module_imports:
            for imp in module_imports[u]:
                raw_imp = strip_prefix(imp)
                if raw_imp in exports_map:
                    for export_m, export_full in exports_map[raw_imp]:
                        if export_m == v:
                            u_left, u_right, u_y = child_box_coords[u][imp]
                            v_left, v_right, v_y = child_box_coords[v][export_full]
                            
                            arrow = patches.FancyArrowPatch(
                                (u_right, u_y),
                                (v_left, v_y),
                                connectionstyle=EDGE_CONNECTIONSTYLE,
                                color=CHILD_EDGE_COLOR,
                                alpha=CHILD_EDGE_ALPHA,
                                arrowstyle=EDGE_ARROWSTYLE,
                                mutation_scale=CHILD_EDGE_MUTATION_SCALE,
                                zorder=3
                            )
                            ax.add_patch(arrow)
                            child_edge_drawn = True

        if not child_edge_drawn:
            # Fallback to module-level edge
            start_x = cx_u + container_width / 2.0
            start_y = cy_u + h_u / 2.0
            end_x = cx_v - container_width / 2.0
            end_y = cy_v + h_v / 2.0
            
            arrow = patches.FancyArrowPatch(
                (start_x, start_y),
                (end_x, end_y),
                connectionstyle=EDGE_CONNECTIONSTYLE,
                color=MODULE_EDGE_COLOR,
                alpha=MODULE_EDGE_ALPHA,
                arrowstyle=EDGE_ARROWSTYLE,
                mutation_scale=MODULE_EDGE_MUTATION_SCALE,
                zorder=3
            )
            ax.add_patch(arrow)



    # ---- 6. Axes limits & title ----
    if mod_center:
        min_x = min(cx - container_width / 2.0 for cx, _ in mod_center.values())
        max_x = max(cx + container_width / 2.0 for cx, _ in mod_center.values())
        min_y = min(cy - container_height[m] / 2.0 for m, (cx, cy) in mod_center.items())
        max_y = max(cy + container_height[m] / 2.0 for m, (cx, cy) in mod_center.items())

        horiz_pad = HORIZ_PAD
        vert_pad = VERT_PAD
        ax.set_xlim(min_x - horiz_pad, max_x + horiz_pad)
        ax.set_ylim(min_y - vert_pad, max_y + vert_pad)

    ax.set_title(PLOT_TITLE)
    ax.axis("off")

    # ---- 7. Add Legend ----
    legend_elements = [
        patches.Patch(facecolor=CONTAINER_FACECOLOR, edgecolor=CONTAINER_EDGECOLOR, label='Module Box'),
        patches.Patch(facecolor=CHILD_FACECOLOR, edgecolor=CHILD_EDGECOLOR, label='Element Box'),
        Line2D([0], [0], color=MODULE_EDGE_COLOR, lw=2, label='Module-to-Module Import'),
        Line2D([0], [0], color=CHILD_EDGE_COLOR, lw=2, label='Specific Element Import'),
        Line2D([0], [0], color='none', label='Format: ClassName (Class)'),
        Line2D([0], [0], color='none', label='Format: <<Interface>> Name (Abstract Class)'),
        Line2D([0], [0], color='none', label='Format: function_name() (Function)'),
        Line2D([0], [0], color='none', label='Format: variable_name (Global Variable)')
    ]
    ax.legend(
        handles=legend_elements, 
        loc='upper left', 
        bbox_to_anchor=(1.02, 1), 
        title="Legend", 
        labelspacing=0.8, 
        fontsize=9, 
        title_fontsize=11
    )
    
    # We restrict the layout from going 100% all the way to the right side of the figure
    # and cap it at 85%. This leaves a physical 15% gutter perfectly sized for the legend, 
    # preventing window overlap or cropping no matter how large the graph scales!
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    if output:
        # small pad; left margin is enforced in data coordinates
        plt.savefig(output, bbox_inches="tight", pad_inches=0.1)
        print(f"Module dependency graph saved to {output}")
    else:
        plt.show()


# ---------------------------
# MAIN
# ---------------------------

def run():
    if PROJECT_ROOT is not None:
        project_root = Path(PROJECT_ROOT).resolve()
    else:
        project_root = Path(__file__).resolve().parent

    print(f"Analyzing project at: {project_root}")

    graph, module_classes, module_functions, module_variables, module_imports = build_module_graph(project_root)
    print(f"Modules: {graph.number_of_nodes()}, Module edges: {len([e for e in graph.edges(data=True) if e[2].get('kind') == 'import'])}")

    output_path = Path(OUTPUT_IMAGE).resolve() if OUTPUT_IMAGE else None
    visualize_module_dependencies(graph, module_classes, module_functions, module_variables, module_imports, output=output_path)


if __name__ == "__main__":
    run()