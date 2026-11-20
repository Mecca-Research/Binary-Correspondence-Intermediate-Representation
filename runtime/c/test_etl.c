/*===- test_etl.c - the C event transducer's own contracts ------------------===
 * PORT-4: bcir_etl.c's contracts the oracle has no counterpart of -- every output buffer short by one, every NULL --
 * beside a golden parse, a golden run and each refusal's element and field. The corpus comparison with the oracle
 * is bcir/tests/test_c_etl.py's, through bcir-etl. Prints OK. */
#include <stdio.h>
#include <string.h>
#include "bcir_etl.h"

static int bad;
#define CHECK(c) do{ if(!(c)){ printf("FAIL %s:%d %s\n",__FILE__,__LINE__,#c); bad++; } }while(0)
static bcir_etl_str S(const char *z){ bcir_etl_str s={z,strlen(z)}; return s; }

static void rop(void){
  static const char text[]="claim add { reads A, B; writes C; count 1024; lane u; stride unit }\nclaim m{reads C}";
  bcir_rop_resource res[8]; bcir_rop_claim cl[4]; uint32_t rids[8];
  bcir_rop_out o={res,8,0,cl,4,0,rids,8,0,0,0,0,0,0};
  CHECK(bcir_rop_parse(text,sizeof text-1,&o)==BCIR_ETL_OK);
  CHECK(o.n_res==3 && res[0].rid==10 && res[2].rid==12 && o.n_claims==2 && o.n_rids==4);
  CHECK(cl[0].id==1000 && cl[0].n_rd==2 && rids[cl[0].rd_first]==10 && rids[cl[0].rd_first+1]==11);
  CHECK(cl[0].n_wr==1 && rids[cl[0].wr_first]==12 && cl[0].count==1024 && cl[0].lane==0 && cl[0].stride==1);
  CHECK(cl[1].id==1001 && cl[1].n_rd==1 && rids[cl[1].rd_first]==12 && cl[1].count==1);
  /* each output one short of what the program needs: a capacity, never a write past it */
  for(int which=0;which<3;which++){
    bcir_rop_out s=o; s.cap_res=which==0?2:8; s.cap_claims=which==1?1:4; s.cap_rids=which==2?3:8;
    CHECK(bcir_rop_parse(text,sizeof text-1,&s)==BCIR_ETL_ERR_CAPACITY); }
  bcir_rop_out z={NULL,0,0,NULL,0,0,NULL,0,0,0,0,0,0,0};
  CHECK(bcir_rop_parse(NULL,1,&z)==BCIR_ETL_ERR_INVALID);
  CHECK(bcir_rop_parse(text,sizeof text-1,NULL)==BCIR_ETL_ERR_INVALID);
  CHECK(bcir_rop_parse("",0,&z)==BCIR_ETL_OK && z.n_claims==0);         /* an empty program: no claims */
  CHECK(bcir_rop_parse("claim a {}",10,&z)==BCIR_ETL_ERR_CAPACITY);      /* a claim, and no room for it */
  /* the refusals, each at the offset the oracle names */
  struct { const char *t; bcir_etl_status st; size_t pos; } r[]={
    {"claim a { # }",BCIR_ETL_ERR_LEX,10}, {"clam a {}",BCIR_ETL_ERR_EXPECT,0}, {"claim {",BCIR_ETL_ERR_EXPECT,6},
    {"claim a { count x }",BCIR_ETL_ERR_EXPECT,16}, {"claim a { reads A B }",BCIR_ETL_ERR_SEPARATOR,18},
    {"claim a { foo }",BCIR_ETL_ERR_STATEMENT,10}, {"claim a { lane q }",BCIR_ETL_ERR_LANE,15},
    {"claim a { stride q }",BCIR_ETL_ERR_STRIDE,17}, {"claim a { count 9223372036854775808 }",BCIR_ETL_ERR_COUNT,16},
    {"claim a { lane u; lane h }",BCIR_ETL_ERR_REPEATED,18}, {"claim a { reads",BCIR_ETL_ERR_EXPECT,15},
  };
  for(size_t k=0;k<sizeof r/sizeof r[0];k++){ bcir_rop_out s=o;
    CHECK(bcir_rop_parse(r[k].t,strlen(r[k].t),&s)==r[k].st && s.pos==r[k].pos); }
  bcir_rop_out s=o;
  CHECK(bcir_rop_parse("claim a { count 9223372036854775807 }",37,&s)==BCIR_ETL_OK && cl[0].count==9223372036854775807ull);
}

static void fsm(void){
  bcir_fsm_state st[]={{S("s0"),0,0},{S("s1"),0,0},{S("acc"),1,0},{S("bad"),0,1}};
  bcir_fsm_transition tx[]={{S("s0"),S("s1"),S("claim"),S("begin")},{S("s1"),S("acc"),S("ident"),S("name")},
                            {S("s1"),S("bad"),S("*"),{NULL,0}},{S("bad"),S("s0"),S("x"),{NULL,0}}};
  uint32_t sx[4], dx[4]; bcir_fsm m; size_t where; int field;
  CHECK(bcir_fsm_compile(&m,st,4,tx,4,S("s0"),sx,dx,&where,&field)==BCIR_ETL_OK && m.start==0);
  bcir_etl_str in[]={S("claim"),S("ident")}; bcir_fsm_step tr[4]; size_t n, fin; int acc;
  CHECK(bcir_fsm_run(&m,in,2,tr,4,&n,&fin,&acc)==BCIR_ETL_OK && acc==1 && fin==2 && n==2 && tr[1].tx==1 && tr[1].sym==1);
  CHECK(bcir_fsm_run(&m,in,2,tr,1,&n,&fin,&acc)==BCIR_ETL_ERR_CAPACITY);   /* a trace one step short */
  bcir_etl_str other[]={S("claim"),S("zzz"),S("x")};                        /* the catch-all to an error state, */
  CHECK(bcir_fsm_run(&m,other,3,tr,4,&n,&fin,&acc)==BCIR_ETL_OK && acc==0 && fin==3 && n==2);   /* which stops it */
  CHECK(bcir_fsm_run(&m,other,0,tr,0,&n,&fin,&acc)==BCIR_ETL_OK && n==0 && fin==0 && acc==0);
  CHECK(bcir_fsm_run(&m,NULL,1,tr,4,&n,&fin,&acc)==BCIR_ETL_ERR_INVALID);
  /* each refusal, naming its element and field */
  bcir_fsm_state dup[]={{S("a"),0,0},{S("a"),0,0}}, both[]={{S("a"),1,1}}, empty[]={{S(""),0,0}};
  CHECK(bcir_fsm_compile(&m,dup,2,NULL,0,S("a"),sx,dx,&where,&field)==BCIR_ETL_ERR_STATE_DUP && where==1);
  CHECK(bcir_fsm_compile(&m,both,1,NULL,0,S("a"),sx,dx,&where,&field)==BCIR_ETL_ERR_STATE_KIND && where==0);
  CHECK(bcir_fsm_compile(&m,empty,1,NULL,0,S(""),sx,dx,&where,&field)==BCIR_ETL_ERR_STATE_NAME);
  bcir_fsm_transition t2[]={{S("s0"),S("s1"),S("a"),{NULL,0}},{S("s0"),S(""),S("a"),{NULL,0}}};
  CHECK(bcir_fsm_compile(&m,st,4,t2,2,S("s0"),sx,dx,&where,&field)==BCIR_ETL_ERR_TX_FIELD && where==1 && field==1);
  bcir_fsm_transition t3[]={{S("s0"),S("nope"),S("a"),{NULL,0}}};
  CHECK(bcir_fsm_compile(&m,st,4,t3,1,S("s0"),sx,dx,&where,&field)==BCIR_ETL_ERR_TX_ENDPOINT && where==0 && field==1);
  bcir_fsm_transition t4[]={{S("s0"),S("s1"),S("a"),{NULL,0}},{S("s0"),S("acc"),S("a"),{NULL,0}}};
  CHECK(bcir_fsm_compile(&m,st,4,t4,2,S("s0"),sx,dx,&where,&field)==BCIR_ETL_ERR_TX_DUP && where==1);
  CHECK(bcir_fsm_compile(&m,st,4,tx,4,S("nowhere"),sx,dx,&where,&field)==BCIR_ETL_ERR_START);
  CHECK(bcir_fsm_compile(&m,st,4,tx,4,S("s0"),NULL,dx,&where,&field)==BCIR_ETL_ERR_INVALID);
}

static void events(void){
  CHECK(bcir_event_stream_check(S("uart0"),S("driver"),S("bytes"),8,4096)==BCIR_ETL_OK);
  CHECK(bcir_event_stream_check(S(""),S("text"),S("utf8"),8,1)==BCIR_ETL_ERR_STREAM_NAME);
  CHECK(bcir_event_stream_check(S("s"),S("video"),S("utf8"),8,1)==BCIR_ETL_ERR_STREAM_KIND);
  CHECK(bcir_event_stream_check(S("s"),S("text"),S("utf16"),8,1)==BCIR_ETL_ERR_STREAM_ENCODING);
  CHECK(bcir_event_stream_check(S("s"),S("text"),S("utf8"),0,1)==BCIR_ETL_ERR_STREAM_BITS);
  CHECK(bcir_event_stream_check(S("s"),S("text"),S("utf8"),8,-1)==BCIR_ETL_ERR_STREAM_WINDOW);
  CHECK(bcir_event_kind_check(S("rx"))==BCIR_ETL_OK && bcir_event_kind_check(S(""))==BCIR_ETL_ERR_KIND_NAME);
  CHECK(bcir_event_check(S("rx"),0,0)==BCIR_ETL_OK && bcir_event_check(S("rx"),3,2)==BCIR_ETL_ERR_EVENT_SPAN);
  CHECK(bcir_event_check(S("rx"),-1,2)==BCIR_ETL_ERR_EVENT_SPAN && bcir_event_check(S(""),0,0)==BCIR_ETL_ERR_EVENT_KIND);
  bcir_etl_str null={NULL,1};
  CHECK(bcir_event_kind_check(null)==BCIR_ETL_ERR_INVALID);
}

int main(void){
  rop(); fsm(); events();
  if(bad) return 1;
  printf("OK\n");
  return 0;
}
