// Minimal illustrative TableGen target-description fragment.
// This is not a complete LLVM target. It demonstrates class/def/let/multiclass
// structure and the kinds of records a real backend expands dramatically.

class ToyReg<string asmName, int enc> {
  string AsmName = asmName;
  int Encoding = enc;
}

def R0 : ToyReg<"r0", 0>;
def R1 : ToyReg<"r1", 1>;

class ToyInst<string mnemonic, int opcode> {
  string Mnemonic = mnemonic;
  int Opcode = opcode;
  bit isCommutable = 0;
  bit mayLoad = 0;
  bit mayStore = 0;
}

let isCommutable = 1 in
def ADDrr : ToyInst<"add", 0x10>;

def LOADri : ToyInst<"load", 0x20> {
  let mayLoad = 1;
}

multiclass BinaryALU<string mnemonic, int baseOpcode> {
  def rr : ToyInst<mnemonic # "rr", baseOpcode>;
  def ri : ToyInst<mnemonic # "ri", !add(baseOpcode, 1)>;
}

defm SUB : BinaryALU<"sub", 0x30>;
