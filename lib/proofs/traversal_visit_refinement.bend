# Visit semantics use an independent context transition and element function.
# The concrete cursor proof supplies both; no caller-supplied read premise is
# needed by the concrete tensor theorems.
import Base
import ../traversal_array.bend as Traversal
import ../traversal_loop.bend as Loop
import ./loop_projection.bend as Projection
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws
import ./traversal_array_refinement.bend as ArrayLaws

def represent(~Reader: Type,~Model: Data,~Element: Data,~Context: Data,~Coordinates: Data,
  ~pack_reader: Model -> Reader,~pack_context: Coordinates -> Context,
  state: Traversal.State<&2,&2,Model,Tree.Tree<Element>,Coordinates>) -> Traversal.State<&1,&1,Reader,Array<Element>,Context>:
  Traversal.State{reader,output,position,coordinates} = state
  Traversal.State{pack_reader(reader),Tree.pack(Element,output),position,pack_context(coordinates)}

def model_step(~Model: Data,~Element: Data,~Coordinates: Data,
  ~sample: Model -> Coordinates -> Element,~next: Coordinates -> Coordinates,
  state: Traversal.State<&2,&2,Model,Tree.Tree<Element>,Coordinates>) -> Traversal.State<&2,&2,Model,Tree.Tree<Element>,Coordinates>:
  Traversal.State{+reader,output,+position,+coordinates} = state
  Traversal.State{reader,Tree.write(Element,output,position,sample(reader,coordinates)),(position + 1 : U32),next(coordinates)}

def step(~Reader: Type,~Model: Data,~Element: Data,~Context: Data,~Coordinates: Data,
  ~pack_reader: Model -> Reader,~pack_context: Coordinates -> Context,
  ~read: Reader -> Context -> Reader & Traversal.Visited<Context,Element>,
  ~sample: Model -> Coordinates -> Element,~next: Coordinates -> Coordinates,
  ~read_matches: @+reader: Model -> @+coordinates: Coordinates ->
    {read(pack_reader(reader),pack_context(coordinates)) == (pack_reader(reader),Traversal.Visited{pack_context(next(coordinates)),sample(reader,coordinates)}) : Reader & Traversal.Visited<Context,Element>},
  +state: Traversal.State<&2,&2,Model,Tree.Tree<Element>,Coordinates>) ->
  {Traversal.visit_step(~Reader,~Element,~Context,~read,represent(~Reader,~Model,~Element,~Context,~Coordinates,~pack_reader,~pack_context,state)) ==
    represent(~Reader,~Model,~Element,~Context,~Coordinates,~pack_reader,~pack_context,model_step(~Model,~Element,~Coordinates,~sample,~next,state)) : Traversal.State<&1,&1,Reader,Array<Element>,Context>}:
  Traversal.State{+reader,+output,+position,+coordinates} = state
  %Equal.sym(Reader & Traversal.Visited<Context,Element>,read(pack_reader(reader),pack_context(coordinates)),
    (pack_reader(reader),Traversal.Visited{pack_context(next(coordinates)),sample(reader,coordinates)}),read_matches(reader,coordinates)) :
    {Traversal.visit_store(~Reader,~Element,~Context,position,Tree.pack(Element,output),_) ==
      represent(~Reader,~Model,~Element,~Context,~Coordinates,~pack_reader,~pack_context,
        model_step(~Model,~Element,~Coordinates,~sample,~next,Traversal.State{reader,output,position,coordinates})) : Traversal.State<&1,&1,Reader,Array<Element>,Context>}
  %Equal.sym(Array<Element>,Array.set(Element,Tree.pack(Element,output),position,sample(reader,coordinates)),
    Tree.pack(Element,Tree.write(Element,output,position,sample(reader,coordinates))),StorageLaws.write(Element,output,position,sample(reader,coordinates))) :
    {Traversal.State{pack_reader(reader),_,(position + 1 : U32),pack_context(next(coordinates))} ==
      represent(~Reader,~Model,~Element,~Context,~Coordinates,~pack_reader,~pack_context,
        model_step(~Model,~Element,~Coordinates,~sample,~next,Traversal.State{reader,output,position,coordinates})) : Traversal.State<&1,&1,Reader,Array<Element>,Context>}
  {==}

def visit(~Reader: Type,~Model: Data,~Element: Data,~Context: Data,~Coordinates: Data,
  ~pack_reader: Model -> Reader,~pack_context: Coordinates -> Context,
  ~read: Reader -> Context -> Reader & Traversal.Visited<Context,Element>,
  ~sample: Model -> Coordinates -> Element,~next: Coordinates -> Coordinates,
  ~read_matches: @+reader: Model -> @+coordinates: Coordinates ->
    {read(pack_reader(reader),pack_context(coordinates)) == (pack_reader(reader),Traversal.Visited{pack_context(next(coordinates)),sample(reader,coordinates)}) : Reader & Traversal.Visited<Context,Element>},
  +count: Nat,+state: Traversal.State<&2,&2,Model,Tree.Tree<Element>,Coordinates>) ->
  {Traversal.visit(~Reader,~Element,~Context,~read,count,represent(~Reader,~Model,~Element,~Context,~Coordinates,~pack_reader,~pack_context,state)) ==
    represent(~Reader,~Model,~Element,~Context,~Coordinates,~pack_reader,~pack_context,
      Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Coordinates>,~model_step(~Model,~Element,~Coordinates,~sample,~next),count,state)) : Traversal.State<&1,&1,Reader,Array<Element>,Context>}:
  Projection.iteration_refines_model(~Traversal.State<&1,&1,Reader,Array<Element>,Context>,~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Coordinates>,
    ~Traversal.visit_step(~Reader,~Element,~Context,~read),~model_step(~Model,~Element,~Coordinates,~sample,~next),
    ~represent(~Reader,~Model,~Element,~Context,~Coordinates,~pack_reader,~pack_context),
    ~step(~Reader,~Model,~Element,~Context,~Coordinates,~pack_reader,~pack_context,~read,~sample,~next,~read_matches),count,state)

def model_update_step(~Model: Data,~Element: Data,~Coordinates: Data,
  ~sample: Model -> Coordinates -> Element,~next: Coordinates -> Coordinates,~combine: Element -> Element -> Element,
  state: Traversal.State<&2,&2,Model,Tree.Tree<Element>,Coordinates>) -> Traversal.State<&2,&2,Model,Tree.Tree<Element>,Coordinates>:
  Traversal.State{+reader,+output,+position,+coordinates} = state
  Traversal.State{reader,Tree.write(Element,output,position,combine(Tree.read(Element,output,position),sample(reader,coordinates))),
    (position + 1 : U32),next(coordinates)}

def update_step(~Reader: Type,~Model: Data,~Element: Data,~Context: Data,~Coordinates: Data,
  ~pack_reader: Model -> Reader,~pack_context: Coordinates -> Context,
  ~read: Reader -> Context -> Reader & Traversal.Visited<Context,Element>,
  ~sample: Model -> Coordinates -> Element,~next: Coordinates -> Coordinates,~combine: Element -> Element -> Element,
  ~read_matches: @+reader: Model -> @+coordinates: Coordinates ->
    {read(pack_reader(reader),pack_context(coordinates)) == (pack_reader(reader),Traversal.Visited{pack_context(next(coordinates)),sample(reader,coordinates)}) : Reader & Traversal.Visited<Context,Element>},
  +state: Traversal.State<&2,&2,Model,Tree.Tree<Element>,Coordinates>) ->
  {Traversal.visit_update_step(~Reader,~Element,~Context,~read,~combine,represent(~Reader,~Model,~Element,~Context,~Coordinates,~pack_reader,~pack_context,state)) ==
    represent(~Reader,~Model,~Element,~Context,~Coordinates,~pack_reader,~pack_context,model_update_step(~Model,~Element,~Coordinates,~sample,~next,~combine,state)) : Traversal.State<&1,&1,Reader,Array<Element>,Context>}:
  Traversal.State{+reader,+output,+position,+coordinates} = state
  %Equal.sym(Array<Element> & Element,Array.get(Element,Tree.pack(Element,output),position),
    (Tree.pack(Element,output),Tree.read(Element,output,position)),StorageLaws.read(Element,output,position)) :
    {Traversal.visit_update_current(~Reader,~Element,~Context,~read,~combine,position,pack_context(coordinates),pack_reader(reader),_) ==
      represent(~Reader,~Model,~Element,~Context,~Coordinates,~pack_reader,~pack_context,
        model_update_step(~Model,~Element,~Coordinates,~sample,~next,~combine,Traversal.State{reader,output,position,coordinates})) : Traversal.State<&1,&1,Reader,Array<Element>,Context>}
  %Equal.sym(Reader & Traversal.Visited<Context,Element>,read(pack_reader(reader),pack_context(coordinates)),
    (pack_reader(reader),Traversal.Visited{pack_context(next(coordinates)),sample(reader,coordinates)}),read_matches(reader,coordinates)) :
    {Traversal.visit_update_sample(~Reader,~Element,~Context,~combine,position,Tree.pack(Element,output),Tree.read(Element,output,position),_) ==
      represent(~Reader,~Model,~Element,~Context,~Coordinates,~pack_reader,~pack_context,
        model_update_step(~Model,~Element,~Coordinates,~sample,~next,~combine,Traversal.State{reader,output,position,coordinates})) : Traversal.State<&1,&1,Reader,Array<Element>,Context>}
  %Equal.sym(Array<Element>,Array.set(Element,Tree.pack(Element,output),position,combine(Tree.read(Element,output,position),sample(reader,coordinates))),
    Tree.pack(Element,Tree.write(Element,output,position,combine(Tree.read(Element,output,position),sample(reader,coordinates)))),
    StorageLaws.write(Element,output,position,combine(Tree.read(Element,output,position),sample(reader,coordinates)))) :
    {Traversal.State{pack_reader(reader),_,(position + 1 : U32),pack_context(next(coordinates))} ==
      represent(~Reader,~Model,~Element,~Context,~Coordinates,~pack_reader,~pack_context,
        model_update_step(~Model,~Element,~Coordinates,~sample,~next,~combine,Traversal.State{reader,output,position,coordinates})) : Traversal.State<&1,&1,Reader,Array<Element>,Context>}
  {==}

def update(~Reader: Type,~Model: Data,~Element: Data,~Context: Data,~Coordinates: Data,
  ~pack_reader: Model -> Reader,~pack_context: Coordinates -> Context,
  ~read: Reader -> Context -> Reader & Traversal.Visited<Context,Element>,
  ~sample: Model -> Coordinates -> Element,~next: Coordinates -> Coordinates,~combine: Element -> Element -> Element,
  ~read_matches: @+reader: Model -> @+coordinates: Coordinates ->
    {read(pack_reader(reader),pack_context(coordinates)) == (pack_reader(reader),Traversal.Visited{pack_context(next(coordinates)),sample(reader,coordinates)}) : Reader & Traversal.Visited<Context,Element>},
  +count: Nat,+state: Traversal.State<&2,&2,Model,Tree.Tree<Element>,Coordinates>) ->
  {Traversal.visit_update(~Reader,~Element,~Context,~read,~combine,count,represent(~Reader,~Model,~Element,~Context,~Coordinates,~pack_reader,~pack_context,state)) ==
    represent(~Reader,~Model,~Element,~Context,~Coordinates,~pack_reader,~pack_context,
      Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Coordinates>,~model_update_step(~Model,~Element,~Coordinates,~sample,~next,~combine),count,state)) : Traversal.State<&1,&1,Reader,Array<Element>,Context>}:
  Projection.iteration_refines_model(~Traversal.State<&1,&1,Reader,Array<Element>,Context>,~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Coordinates>,
    ~Traversal.visit_update_step(~Reader,~Element,~Context,~read,~combine),~model_update_step(~Model,~Element,~Coordinates,~sample,~next,~combine),
    ~represent(~Reader,~Model,~Element,~Context,~Coordinates,~pack_reader,~pack_context),
    ~update_step(~Reader,~Model,~Element,~Context,~Coordinates,~pack_reader,~pack_context,~read,~sample,~next,~combine,~read_matches),count,state)

