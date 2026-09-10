// The C++ object model as the frontend materializes it.
// Companion chapter: training/llvm/20-clang-frontend/04-cxx-lowering-rules.md
//
//   clang++ -O0 -S -emit-llvm -fno-discard-value-names cxx-object-model.cpp -o -
//   clang++ -Xclang -fdump-vtable-layouts -c cxx-object-model.cpp -o /dev/null

struct Shape {
  virtual ~Shape();
  virtual int area() const = 0;
  virtual int perimeter() const { return 0; }
};

struct Square : Shape {
  int side;
  explicit Square(int s) : side(s) {}
  ~Square() override;
  int area() const override { return side * side; }
};

int total_area(const Shape &s) { return s.area(); }
int square_area(const Square &s) { return s.area(); }

// Overloads and templates: one C++ name, several IR symbols.
int scale(int v) { return v * 2; }
double scale(double v) { return v * 2.0; }

template <typename T> T twice(T v) { return v + v; }
int use_template() { return twice<int>(21); }

// A temporary with a non-trivial destructor pins its lifetime to the full
// expression; the frontend, not LLVM, decides where the destructor call goes.
struct Guard { Guard(); ~Guard(); int n; };
int guarded() { return Guard().n; }
