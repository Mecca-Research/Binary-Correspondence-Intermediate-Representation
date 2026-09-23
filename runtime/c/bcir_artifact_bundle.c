/*===- bcir_artifact_bundle.c - allocation-free BCAB v1 trust boundary ----===*/
#include "bcir_artifact_bundle.h"

#include "bcir_runtime.h"
#include "bcir_sha256.h"

#define AB_ENTRY_FLAGS 0x0fu

static uint16_t rd16(const uint8_t *p) {
  return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}
static uint32_t rd32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
         ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static uint64_t rd64(const uint8_t *p) {
  return (uint64_t)rd32(p) | ((uint64_t)rd32(p + 4) << 32);
}
static uint32_t rd32be(const uint8_t *p) {
  return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
         ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}
static uint16_t rd16be(const uint8_t *p) {
  return (uint16_t)(((uint16_t)p[0] << 8) | (uint16_t)p[1]);
}
static void zero_bytes(void *pointer, size_t length) {
  uint8_t *p = (uint8_t *)pointer;
  for (size_t i = 0; i < length; ++i) p[i] = 0;
}
static void copy_bytes(void *destination, const void *source, size_t length) {
  uint8_t *d = (uint8_t *)destination;
  const uint8_t *s = (const uint8_t *)source;
  for (size_t i = 0; i < length; ++i) d[i] = s[i];
}
static int equal_bytes(const void *lhs, const void *rhs, size_t length) {
  const uint8_t *a = (const uint8_t *)lhs, *b = (const uint8_t *)rhs;
  uint8_t difference = 0;
  for (size_t i = 0; i < length; ++i) difference |= (uint8_t)(a[i] ^ b[i]);
  return difference == 0;
}
static int all_zero(const uint8_t *data, size_t length) {
  uint8_t aggregate = 0;
  for (size_t i = 0; i < length; ++i) aggregate |= data[i];
  return aggregate == 0;
}
static uint64_t align8(uint64_t value) { return (value + 7u) & ~UINT64_C(7); }

/* SHA-256 is the shared freestanding unit (bcir_sha256.c): it began here, file-local, and
 * moved verbatim when the control plane (G14) needed the same digest and a MAC over it. */

static int string_compare(const char *a, const char *b) {
  while (*a && *a == *b) { ++a; ++b; }
  return (unsigned char)*a < (unsigned char)*b ? -1 :
         ((unsigned char)*a > (unsigned char)*b ? 1 : 0);
}
static int string_equal(const char *a, const char *b) { return string_compare(a, b) == 0; }

static int bounded_caller_string(const char *value, size_t maximum) {
  if (!value) return 1;
  size_t end = 0;
  while (end <= maximum && value[end] != '\0') ++end;
  if (end > maximum || (end && (value[0] == ' ' || value[end-1] == ' '))) return 0;
  for (size_t i = 0; i < end; ++i)
    if ((unsigned char)value[i] < 0x20u || (unsigned char)value[i] > 0x7eu) return 0;
  return 1;
}

static int fixed_string(const uint8_t *wire, size_t width, char *out, int required) {
  size_t end = 0;
  while (end < width && wire[end] != 0) ++end;
  if (end == width || (required && end == 0)) return 0;
  if (end && (wire[0] == ' ' || wire[end-1] == ' ')) return 0;
  for (size_t i = 0; i < end; ++i)
    if (wire[i] < 0x20u || wire[i] > 0x7eu) return 0;
  for (size_t i = end; i < width; ++i) if (wire[i] != 0) return 0;
  for (size_t i = 0; i < end; ++i) out[i] = (char)wire[i];
  out[end] = '\0';
  return 1;
}
static int feature_char(char c) {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
         (c >= '0' && c <= '9') || c == '_' || c == '.' || c == '+' ||
         c == ':' || c == '-';
}
static int token_compare(const char *a, size_t an, const char *b, size_t bn) {
  size_t n = an < bn ? an : bn;
  for (size_t i = 0; i < n; ++i)
    if ((unsigned char)a[i] != (unsigned char)b[i])
      return (unsigned char)a[i] < (unsigned char)b[i] ? -1 : 1;
  return an < bn ? -1 : (an > bn ? 1 : 0);
}
static int valid_feature_csv(const char *csv) {
  if (!csv || !*csv) return csv != NULL;
  const char *previous = NULL; size_t previous_length = 0;
  const char *cursor = csv;
  while (*cursor) {
    const char *start = cursor;
    while (*cursor && *cursor != ',') {
      if (!feature_char(*cursor)) return 0;
      ++cursor;
    }
    size_t length = (size_t)(cursor - start);
    if (!length || (previous && token_compare(previous, previous_length, start, length) >= 0)) return 0;
    previous = start; previous_length = length;
    if (*cursor == ',') { ++cursor; if (!*cursor) return 0; }
  }
  return 1;
}
static int csv_has(const char *csv, const char *token, size_t length) {
  if (!csv) return 0;
  const char *cursor = csv;
  while (*cursor) {
    const char *start = cursor;
    while (*cursor && *cursor != ',') ++cursor;
    size_t current = (size_t)(cursor - start);
    int comparison = token_compare(start, current, token, length);
    if (comparison == 0) return 1;
    if (comparison > 0) return 0;
    if (*cursor == ',') ++cursor;
  }
  return 0;
}
static int csv_subset(const char *required, const char *available) {
  const char *cursor = required;
  while (cursor && *cursor) {
    const char *start = cursor;
    while (*cursor && *cursor != ',') ++cursor;
    if (!csv_has(available, start, (size_t)(cursor-start))) return 0;
    if (*cursor == ',') ++cursor;
  }
  return 1;
}
static int csv_disjoint(const char *lhs, const char *rhs) {
  const char *cursor = lhs;
  while (cursor && *cursor) {
    const char *start = cursor;
    while (*cursor && *cursor != ',') ++cursor;
    if (csv_has(rhs, start, (size_t)(cursor-start))) return 0;
    if (*cursor == ',') ++cursor;
  }
  return 1;
}
static unsigned csv_count(const char *csv) {
  unsigned count = 0;
  if (csv && *csv) { count = 1; for (; *csv; ++csv) if (*csv == ',') ++count; }
  return count;
}

static int valid_utf8_text(const uint8_t *data, size_t length) {
  size_t i = 0;
  while (i < length) {
    uint8_t c = data[i++];
    if (c == 0) return 0;
    if (c < 0x80u) continue;
    unsigned need; uint32_t value, minimum;
    if ((c & 0xe0u) == 0xc0u) { need=1; value=c&0x1fu; minimum=0x80u; }
    else if ((c & 0xf0u) == 0xe0u) { need=2; value=c&0x0fu; minimum=0x800u; }
    else if ((c & 0xf8u) == 0xf0u) { need=3; value=c&0x07u; minimum=0x10000u; }
    else return 0;
    if (need > length-i) return 0;
    for (unsigned j=0; j<need; ++j) {
      uint8_t continuation=data[i++]; if ((continuation&0xc0u)!=0x80u) return 0;
      value=(value<<6)|(continuation&0x3fu);
    }
    if (value<minimum || value>0x10ffffu || (value>=0xd800u && value<=0xdfffu)) return 0;
  }
  return 1;
}
static int contains_bytes(const uint8_t *data, size_t length, const char *needle) {
  size_t n = 0; while (needle[n]) ++n;
  if (n > length) return 0;
  for (size_t i=0; i<=length-n; ++i) if (equal_bytes(data+i, needle, n)) return 1;
  return 0;
}

static int kind_format(uint16_t kind, uint16_t format) {
  static const uint8_t formats[26] = {
    0,1,2,2,3,4,5,6,7,8,8,2,9,10,8,8,8,8,8,2,11,11,4,4,12,13
  };
  return kind >= 1u && kind <= (uint16_t)BCIR_AB_EXECUTION_PLAN && formats[kind] == format;
}

static bcir_ab_status decode_entry(const bcir_ab_view *view, uint32_t index,
                                   bcir_ab_entry *out) {
  zero_bytes(out, sizeof(*out));
  if (!view || !view->data || index >= view->count) return BCIR_AB_ERR_INVALID;
  uint64_t offset64 = BCIR_AB_HEADER_SIZE + (uint64_t)index * BCIR_AB_ENTRY_SIZE;
  if (offset64 > view->length || BCIR_AB_ENTRY_SIZE > view->length-(size_t)offset64)
    return BCIR_AB_ERR_LAYOUT;
  const uint8_t *p = view->data + (size_t)offset64;
  out->kind=rd16(p); out->format=rd16(p+2); out->endianness=p[4]; out->pointer_bits=p[5];
  out->flags=rd16(p+6); out->machine=rd32(p+8); out->priority=(int32_t)rd32(p+12);
  out->payload_offset=rd64(p+16); out->payload_size=rd64(p+24);
  out->provenance_digest=rd64(p+32); out->cal_gen=rd64(p+40);
  out->payload_crc32=rd32(p+48);
  if (rd32(p+52) || (out->flags & ~AB_ENTRY_FLAGS) || !kind_format(out->kind,out->format) ||
      out->endianness > BCIR_AB_ENDIAN_BIG ||
      (out->pointer_bits != 0 && out->pointer_bits != 32 && out->pointer_bits != 64))
    return BCIR_AB_ERR_METADATA;
  copy_bytes(out->payload_sha256,p+56,32); copy_bytes(out->target_manifest_sha256,p+88,32);
  if (!fixed_string(p+120,48,out->variant_id,1) || !fixed_string(p+168,48,out->triple,0) ||
      !fixed_string(p+216,24,out->architecture,0) || !fixed_string(p+240,24,out->os_abi,0) ||
      !fixed_string(p+264,24,out->channel,0) || !fixed_string(p+288,32,out->entry_symbol,0) ||
      !fixed_string(p+320,64,out->required_features,0) ||
      !fixed_string(p+384,64,out->prohibited_features,0)) return BCIR_AB_ERR_METADATA;
  if (!valid_feature_csv(out->required_features) || !valid_feature_csv(out->prohibited_features) ||
      !csv_disjoint(out->required_features,out->prohibited_features)) return BCIR_AB_ERR_METADATA;
  if ((out->format==BCIR_AB_FMT_ELF || out->format==BCIR_AB_FMT_COFF ||
       out->format==BCIR_AB_FMT_MACHO || out->format==BCIR_AB_FMT_PE) &&
      (!out->pointer_bits || out->endianness==BCIR_AB_ENDIAN_NEUTRAL || !out->machine))
    return BCIR_AB_ERR_METADATA;
  if ((out->kind==BCIR_AB_ELF_EXECUTABLE || out->kind==BCIR_AB_PE_EXECUTABLE ||
       out->kind==BCIR_AB_MACHO_EXECUTABLE) &&
      !(out->flags&BCIR_AB_FLAG_EXECUTABLE)) return BCIR_AB_ERR_METADATA;
  if (out->payload_offset > view->length || out->payload_size > view->length-(size_t)out->payload_offset)
    return BCIR_AB_ERR_LAYOUT;
  out->payload=view->data+(size_t)out->payload_offset;
  return BCIR_AB_OK;
}

static bcir_ab_status validate_payload(const bcir_ab_entry *entry) {
  const uint8_t *p=entry->payload; size_t n=(size_t)entry->payload_size;
  if (!n) return BCIR_AB_ERR_PAYLOAD;
  if (entry->kind==BCIR_AB_STREAM_PACK)
    return bcir_sp_verify_semantic(p,n,UINT32_MAX,UINT32_MAX)==BCIR_OK ? BCIR_AB_OK : BCIR_AB_ERR_PAYLOAD;
  if (entry->kind==BCIR_AB_EXECUTION_PLAN)   /* the plan as bytes: the full wire-law walk (G11) */
    return bcir_ep_verify(p,n)==BCIR_OK ? BCIR_AB_OK : BCIR_AB_ERR_PAYLOAD;
  if (entry->format==BCIR_AB_FMT_ELF) {
    if (n<20 || !equal_bytes(p,"\x7f" "ELF",4) || (p[4]!=1 && p[4]!=2) || (p[5]!=1 && p[5]!=2))
      return BCIR_AB_ERR_PAYLOAD;
    uint8_t endian=p[5]==1?BCIR_AB_ENDIAN_LITTLE:BCIR_AB_ENDIAN_BIG;
    uint8_t bits=p[4]==1?32:64; uint32_t machine=p[5]==1?rd16(p+18):rd16be(p+18);
    uint16_t elf_type=p[5]==1?rd16(p+16):rd16be(p+16);
    if (entry->endianness!=endian || entry->pointer_bits!=bits ||
        entry->machine!=machine) return BCIR_AB_ERR_PAYLOAD;
    if ((entry->kind==BCIR_AB_ELF_OBJECT && elf_type!=1) ||
        (entry->kind==BCIR_AB_ELF_SHARED && elf_type!=3) ||
        (entry->kind==BCIR_AB_ELF_EXECUTABLE && elf_type!=2 && elf_type!=3))
      return BCIR_AB_ERR_PAYLOAD;
  } else if (entry->format==BCIR_AB_FMT_COFF) {
    if (n<20 || entry->endianness!=BCIR_AB_ENDIAN_LITTLE ||
        entry->machine!=rd16(p)) return BCIR_AB_ERR_PAYLOAD;
  } else if (entry->format==BCIR_AB_FMT_MACHO) {
    if (n<28) return BCIR_AB_ERR_PAYLOAD;
    uint32_t magic=rd32(p); uint8_t endian,bits; uint32_t machine,file_type;
    if (magic==0xfeedfaceu || magic==0xfeedfacfu) {
      endian=BCIR_AB_ENDIAN_LITTLE; bits=magic==0xfeedfacfu?64:32;
      machine=rd32(p+4); file_type=rd32(p+12);
    } else if (rd32be(p)==0xfeedfaceu || rd32be(p)==0xfeedfacfu) {
      uint32_t be=rd32be(p); endian=BCIR_AB_ENDIAN_BIG; bits=be==0xfeedfacfu?64:32;
      machine=rd32be(p+4); file_type=rd32be(p+12);
    } else return BCIR_AB_ERR_PAYLOAD;
    if (entry->endianness!=endian || entry->pointer_bits!=bits ||
        entry->machine!=machine) return BCIR_AB_ERR_PAYLOAD;
    if ((entry->kind==BCIR_AB_MACHO_OBJECT && file_type!=1) ||
        (entry->kind==BCIR_AB_MACHO_EXECUTABLE && file_type!=2) ||
        (entry->kind==BCIR_AB_MACHO_SHARED && file_type!=6)) return BCIR_AB_ERR_PAYLOAD;
  } else if (entry->format==BCIR_AB_FMT_PE) {
    if (n<0x40 || p[0]!='M' || p[1]!='Z') return BCIR_AB_ERR_PAYLOAD;
    uint32_t pe_offset=rd32(p+0x3c);
    if (pe_offset>n-24 || !equal_bytes(p+pe_offset,"PE\0\0",4)) return BCIR_AB_ERR_PAYLOAD;
    uint32_t machine=rd16(p+pe_offset+4), optional_size=rd16(p+pe_offset+20);
    uint16_t characteristics=rd16(p+pe_offset+22);
    if (optional_size<2 || optional_size>n-(pe_offset+24)) return BCIR_AB_ERR_PAYLOAD;
    uint16_t optional_magic=rd16(p+pe_offset+24);
    uint8_t bits=optional_magic==0x10b?32:(optional_magic==0x20b?64:0);
    int is_dll=(characteristics&0x2000u)!=0;
    if (!bits || entry->endianness!=BCIR_AB_ENDIAN_LITTLE || entry->pointer_bits!=bits ||
        entry->machine!=machine ||
        (is_dll!=(entry->kind==BCIR_AB_PE_SHARED))) return BCIR_AB_ERR_PAYLOAD;
  } else if (entry->format==BCIR_AB_FMT_ARCHIVE) {
    if (n<8 || !equal_bytes(p,"!<arch>\n",8)) return BCIR_AB_ERR_PAYLOAD;
  } else if (entry->format==BCIR_AB_FMT_WASM) {
    static const uint8_t magic[8]={0,0x61,0x73,0x6d,1,0,0,0};
    if (n<8 || !equal_bytes(p,magic,8)) return BCIR_AB_ERR_PAYLOAD;
  } else if (entry->format==BCIR_AB_FMT_LLVM_BITCODE) {
    static const uint8_t raw[4]={0x42,0x43,0xc0,0xde}, wrapper[4]={0xde,0xc0,0x17,0x0b};
    if (n<4 || (!equal_bytes(p,raw,4) && !equal_bytes(p,wrapper,4))) return BCIR_AB_ERR_PAYLOAD;
  } else if (entry->format==BCIR_AB_FMT_TEXT) {
    if (!valid_utf8_text(p,n)) return BCIR_AB_ERR_PAYLOAD;
    if (entry->kind==BCIR_AB_PTX &&
        (!contains_bytes(p,n,".version") || !contains_bytes(p,n,".target"))) return BCIR_AB_ERR_PAYLOAD;
  } else if (entry->format==BCIR_AB_FMT_SPIRV) {
    if (n<20 || (n&3u) || (rd32(p)!=0x07230203u && rd32be(p)!=0x07230203u))
      return BCIR_AB_ERR_PAYLOAD;
  } else if (entry->format==BCIR_AB_FMT_JVM_CLASS) {
    if (n<10 || rd32be(p)!=0xcafebabeu) return BCIR_AB_ERR_PAYLOAD;
  } else if (entry->format!=BCIR_AB_FMT_RAW) return BCIR_AB_ERR_PAYLOAD;
  return BCIR_AB_OK;
}

bcir_ab_status bcir_ab_open(const uint8_t *data, size_t length, bcir_ab_view *out) {
  if (out) zero_bytes(out,sizeof(*out));
  if (!data || !out) return BCIR_AB_ERR_INVALID;
  if (length<BCIR_AB_HEADER_SIZE || (uint64_t)length>BCIR_AB_MAX_BYTES) return BCIR_AB_ERR_SIZE;
  if (!equal_bytes(data,BCIR_AB_MAGIC,4)) return BCIR_AB_ERR_MAGIC;
  if (rd16(data+4)!=BCIR_AB_VERSION) return BCIR_AB_ERR_VERSION;
  uint32_t count=rd32(data+12), root=rd32(data+20), def=rd32(data+24);
  uint64_t directory_offset=rd64(data+28), directory_size=rd64(data+36);
  uint64_t payload_offset=rd64(data+44), file_size=rd64(data+52);
  if (rd16(data+6)!=BCIR_AB_HEADER_SIZE || rd32(data+8)!=0 ||
      rd32(data+16)!=BCIR_AB_ENTRY_SIZE || !count || count>BCIR_AB_MAX_ENTRIES ||
      file_size!=(uint64_t)length || directory_offset!=BCIR_AB_HEADER_SIZE ||
      directory_size!=(uint64_t)count*BCIR_AB_ENTRY_SIZE ||
      payload_offset!=align8(directory_offset+directory_size) || payload_offset>length ||
      (root!=BCIR_AB_NO_INDEX && root>=count) || (def!=BCIR_AB_NO_INDEX && def>=count) ||
      !all_zero(data+116,12)) return BCIR_AB_ERR_LAYOUT;
  uint8_t header[BCIR_AB_HEADER_SIZE]; copy_bytes(header,data,sizeof(header));
  zero_bytes(header+80,4);
  if (bcir_crc32(header,sizeof(header))!=rd32(data+80)) return BCIR_AB_ERR_CRC;
  if (bcir_crc32(data+BCIR_AB_HEADER_SIZE,length-BCIR_AB_HEADER_SIZE)!=rd32(data+76))
    return BCIR_AB_ERR_CRC;
  uint8_t actual_sha[32], zeros[36]; bcir_sha256 state;
  zero_bytes(zeros,sizeof(zeros)); bcir_sha256_init(&state); bcir_sha256_update(&state,data,80);
  bcir_sha256_update(&state,zeros,sizeof(zeros)); bcir_sha256_update(&state,data+116,length-116); bcir_sha256_final(&state,actual_sha);
  if (!equal_bytes(actual_sha,data+84,32)) return BCIR_AB_ERR_SHA256;
  uint64_t directory_end=directory_offset+directory_size;
  if (!all_zero(data+(size_t)directory_end,(size_t)(payload_offset-directory_end)))
    return BCIR_AB_ERR_LAYOUT;
  bcir_ab_view candidate;
  zero_bytes(&candidate,sizeof(candidate)); candidate.data=data; candidate.length=length;
  candidate.count=count; candidate.root_index=root; candidate.default_index=def;
  candidate.provenance_digest=rd64(data+60); candidate.generation=rd64(data+68);
  candidate.body_crc32=rd32(data+76); candidate.header_crc32=rd32(data+80);
  copy_bytes(candidate.embedded_sha256,data+84,32);
  uint64_t previous_end=payload_offset; char previous_id[48]; previous_id[0]='\0';
  for (uint32_t i=0; i<count; ++i) {
    bcir_ab_entry entry; bcir_ab_status status=decode_entry(&candidate,i,&entry);
    if (status!=BCIR_AB_OK) return status;
    uint64_t wanted=align8(previous_end);
    if (wanted<previous_end || entry.payload_offset!=wanted || !entry.payload_size ||
        entry.payload_offset>length || entry.payload_size>(uint64_t)length-entry.payload_offset ||
        !all_zero(data+(size_t)previous_end,(size_t)(entry.payload_offset-previous_end)))
      return BCIR_AB_ERR_LAYOUT;
    if (previous_id[0] && string_compare(entry.variant_id,previous_id)<=0) return BCIR_AB_ERR_METADATA;
    copy_bytes(previous_id,entry.variant_id,sizeof(previous_id));
    if (bcir_crc32(entry.payload,(size_t)entry.payload_size)!=entry.payload_crc32) return BCIR_AB_ERR_CRC;
    bcir_sha256_digest(entry.payload,(size_t)entry.payload_size,actual_sha);
    if (!equal_bytes(actual_sha,entry.payload_sha256,32)) return BCIR_AB_ERR_SHA256;
    status=validate_payload(&entry); if (status!=BCIR_AB_OK) return status;
    previous_end=entry.payload_offset+entry.payload_size;
  }
  uint64_t final_end=align8(previous_end);
  if (final_end<previous_end || final_end!=(uint64_t)length ||
      !all_zero(data+(size_t)previous_end,(size_t)(final_end-previous_end))) return BCIR_AB_ERR_LAYOUT;
  if (root!=BCIR_AB_NO_INDEX) {
    bcir_ab_entry entry; if (decode_entry(&candidate,root,&entry)!=BCIR_AB_OK ||
                              entry.kind!=BCIR_AB_STREAM_PACK) return BCIR_AB_ERR_METADATA;
  }
  if (def!=BCIR_AB_NO_INDEX) {
    bcir_ab_entry entry;
    if (decode_entry(&candidate,def,&entry)!=BCIR_AB_OK ||
        ((entry.flags&BCIR_AB_FLAG_EXECUTABLE) &&
         !(entry.flags&BCIR_AB_FLAG_R12_ATTESTED))) return BCIR_AB_ERR_METADATA;
  }
  *out=candidate;
  return BCIR_AB_OK;
}

bcir_ab_status bcir_ab_get(const bcir_ab_view *view, uint32_t index, bcir_ab_entry *out) {
  if (out) zero_bytes(out,sizeof(*out));
  if (!out) return BCIR_AB_ERR_INVALID;
  return decode_entry(view,index,out);
}

static int compatible(const bcir_ab_entry *entry, const bcir_ab_envelope *env) {
  if (env->accepted_kind_mask && !(env->accepted_kind_mask&(UINT64_C(1)<<entry->kind))) return 0;
  if (env->accepted_format_mask && !(env->accepted_format_mask&(UINT64_C(1)<<entry->format))) return 0;
  const char *values[4]={env->triple,env->architecture,env->os_abi,env->channel};
  const char *required[4]={entry->triple,entry->architecture,entry->os_abi,entry->channel};
  for (unsigned i=0;i<4;++i) if (required[i][0] && (!values[i] || !string_equal(required[i],values[i]))) return 0;
  if (entry->endianness!=BCIR_AB_ENDIAN_NEUTRAL && entry->endianness!=env->endianness) return 0;
  if (entry->pointer_bits && entry->pointer_bits!=env->pointer_bits) return 0;
  if (entry->machine && entry->machine!=env->machine) return 0;
  if (!csv_subset(entry->required_features,env->features) ||
      !csv_disjoint(entry->prohibited_features,env->features)) return 0;
  if (!all_zero(entry->target_manifest_sha256,32) &&
      (!env->target_manifest_sha256 ||
       !equal_bytes(entry->target_manifest_sha256,env->target_manifest_sha256,32))) return 0;
  if (entry->cal_gen && (!env->has_cal_gen || entry->cal_gen!=env->cal_gen)) return 0;
  if (env->require_r12 && (entry->flags&BCIR_AB_FLAG_EXECUTABLE) &&
      !(entry->flags&BCIR_AB_FLAG_R12_ATTESTED)) return 0;
  if ((entry->flags&BCIR_AB_FLAG_DEBUG) && !env->allow_debug) return 0;
  return 1;
}
static int valid_envelope(const bcir_ab_envelope *envelope) {
  const uint64_t kind_mask =
      ((UINT64_C(1) << (BCIR_AB_EXECUTION_PLAN + 1u)) - UINT64_C(2));
  const uint64_t format_mask =
      ((UINT64_C(1) << (BCIR_AB_FMT_EXECUTION_PLAN + 1u)) - UINT64_C(1));
  if (!bounded_caller_string(envelope->triple, BCIR_AB_TRIPLE_MAX) ||
      !bounded_caller_string(envelope->architecture, BCIR_AB_TARGET_NAME_MAX) ||
      !bounded_caller_string(envelope->os_abi, BCIR_AB_TARGET_NAME_MAX) ||
      !bounded_caller_string(envelope->channel, BCIR_AB_TARGET_NAME_MAX) ||
      !bounded_caller_string(envelope->features, BCIR_AB_FEATURE_CSV_MAX) ||
      !valid_feature_csv(envelope->features ? envelope->features : "") ||
      (envelope->accepted_kind_mask & ~kind_mask) ||
      (envelope->accepted_format_mask & ~format_mask) ||
      envelope->endianness > BCIR_AB_ENDIAN_BIG ||
      (envelope->pointer_bits != 0 && envelope->pointer_bits != 32 &&
       envelope->pointer_bits != 64) ||
      envelope->has_cal_gen > 1u || envelope->require_r12 > 1u ||
      envelope->allow_debug > 1u ||
      (!envelope->has_cal_gen && envelope->cal_gen != 0))
    return 0;
  return 1;
}
static int specificity(const bcir_ab_entry *entry) {
  int score=0; const char *fields[6]={entry->triple,entry->architecture,entry->os_abi,
    entry->channel,entry->entry_symbol,NULL};
  for (unsigned i=0;i<5;++i) if (fields[i][0]) ++score;
  if (!all_zero(entry->target_manifest_sha256,32)) ++score;
  if (entry->pointer_bits) ++score;
  if (entry->machine) ++score;
  if (entry->endianness!=BCIR_AB_ENDIAN_NEUTRAL) ++score;
  if (entry->cal_gen) ++score;
  return score;
}
bcir_ab_status bcir_ab_select(const bcir_ab_view *view, const bcir_ab_envelope *envelope,
                              const char *requested_id, uint32_t *selected_index) {
  if (selected_index) *selected_index=BCIR_AB_NO_INDEX;
  if (!view || !view->data || !selected_index) return BCIR_AB_ERR_INVALID;
  if (!bounded_caller_string(requested_id, BCIR_AB_VARIANT_ID_MAX) ||
      (envelope && !valid_envelope(envelope))) return BCIR_AB_ERR_INVALID;
  if (requested_id && *requested_id) {
    for (uint32_t i=0;i<view->count;++i) { bcir_ab_entry entry;
      if (decode_entry(view,i,&entry)!=BCIR_AB_OK) return BCIR_AB_ERR_INVALID;
      if (string_equal(entry.variant_id,requested_id)) {
        if (envelope && !compatible(&entry,envelope)) return BCIR_AB_ERR_INCOMPATIBLE;
        *selected_index=i; return BCIR_AB_OK;
      }
    }
    return BCIR_AB_ERR_NOT_FOUND;
  }
  if (!envelope) {
    if (view->default_index==BCIR_AB_NO_INDEX) return BCIR_AB_ERR_NOT_FOUND;
    *selected_index=view->default_index; return BCIR_AB_OK;
  }
  int found=0; uint32_t best_index=0; bcir_ab_entry best;
  for (uint32_t i=0;i<view->count;++i) { bcir_ab_entry entry;
    if (decode_entry(view,i,&entry)!=BCIR_AB_OK) return BCIR_AB_ERR_INVALID;
    if (!compatible(&entry,envelope)) continue;
    if (!found || entry.priority>best.priority ||
        (entry.priority==best.priority && specificity(&entry)>specificity(&best)) ||
        (entry.priority==best.priority && specificity(&entry)==specificity(&best) &&
         csv_count(entry.required_features)>csv_count(best.required_features)) ||
        (entry.priority==best.priority && specificity(&entry)==specificity(&best) &&
         csv_count(entry.required_features)==csv_count(best.required_features) &&
         string_compare(entry.variant_id,best.variant_id)<0)) {
      best=entry; best_index=i; found=1;
    }
  }
  if (!found) return BCIR_AB_ERR_INCOMPATIBLE;
  *selected_index=best_index; return BCIR_AB_OK;
}

const char *bcir_ab_status_string(bcir_ab_status status) {
  static const char *const names[] = {"ok","invalid","size","magic","version","layout",
    "metadata","crc","sha256","payload","incompatible","not-found"};
  return (unsigned)status < sizeof(names)/sizeof(names[0]) ? names[status] : "unknown";
}
