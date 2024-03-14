-- meta=init_sections=create_tables,create_idx

------ DML creates

-- name=create_idx
create index doc_dir_id on doc(dir_id);
create index doc_file on doc(file);
create index sent_doc_id on sent(doc_id);
create index sent_sid on sent(sid);
create index sent_lang on sent(lang);
create index sent_sent on sent(sent);
create index corp_doc_doc_id on corp_doc(doc_id);
create index corp_doc_sid on corp_doc(sid);
create index corp_doc_name on corp_doc(name);
create index corp_doc_file on corp_doc(file);
create index corp_src_id on corp_src(id);
create index corp_src_name on corp_src(name);
create index corp_src_url on corp_src(url);

-- name=create_tables
create table doc (dir_id text, file text);
create table sent (doc_id int, six int, sid text, lang char(2), sent text, spans text);
create table corp_doc (doc_id int, sid int, name text, file text);
create table corp_src (id text, name text, url text);

------ dbutil hooks

---- inserts

-- name=insert_doc
insert into doc (dir_id, file) values (?, ?);

-- name=insert_sent
insert into sent_keep (doc_id, six, sid, lang, sent, spans, toklen) values (?, ?, ?, ?, ?, ?, ?);

-- name=insert_corp_doc
insert into corp_doc (doc_id, sid, name, file) values (?, ?, ?, ?);

-- name=insert_source
insert into corp_src (id, name, url) values (?, ?, ?);


---- deletes

-- name=delete_corp_doc
delete from corp_doc

-- name=delete_corp_src
delete from corp_src


---- selects

-- counts

-- name=doc_count
select count(*) from doc_keep;

-- name=sent_count
select count(*) as cnt
  from sent
  where doc_id in (select doc_id from doc_keep);


-- name=corp_doc_count
select count(*) from corp_doc;

-- name=corp_doc_name_count
select count(distinct(name)) from corp_doc order by name;


-- keys and exists

-- name=select_doc_ids
select doc_id from doc_keep;

-- name=select_doc_exists
select count(*) from doc_keep where doc_id = ?;


-- gets

-- name=select_corpus_source
select cs.name, cs.id, cd.file, d.dir_id as directory_id, d.file
  from corp_src as cs, corp_doc as cd, doc as d
  where cd.name = cs.id and
        cd.doc_id = d.rowid;

-- name=select_doc
select rowid, dir_id, file from doc;

-- name=select_corp_doc_select
select * from corp_doc;

-- name=select_corp_doc_names
select distinct(name) from corp_doc order by name;

-- name=select_sent_by_doc_id
select s.sent, s.spans, s.sid, s.rowid
  from sent as s
  where doc_id = ?
  order by six;


-- joined queries

-- name=select_corp_doc_count_by_name
select name, count(name) as cnt
  from corp_doc
  group by name;

-- name=select_corp_by_doc_count
select *, count(name) as cnt
  from corp_doc
  group by name;


------ reporting only: remaining queries aren't, or won't be added to the code
------ base

-- find sources that have only one document
select *
  from corp_doc group by name having count(name) == 1;

-- get a taste of sentences and their documents
select d.rowid, s.sid, s.sent
  from doc as d, sent as s
  where d.rowid in (select rowid from corp_doc group by name having count(name) == 1) and
        d.rowid = s.doc_id
  limit 50;

-- orphan off the document sample window (cisql only)
orph sent_by_doc

-- get document corpus source urls
select cd.name, cd.file, u.url
  from corp_doc as cd, corp_src cs
  where cs.id = cd.name
  order by cd.name, cd.file;


---- data analysis

-- confirm the assumption the Albanian language code "al" isn't used
-- al: 0, sq: 8,365,699
select count(*) from sent where lang = 'al';
select count(*) from sent where lang = 'sq';

-- add document counts
drop table if exists doc_counts;
create table doc_counts as
  select d.rowid as doc_id, (select count(*) from sent where doc_id = d.rowid) as cnt
    from doc as d;

-- all documents with corups and counts
select d.rowid, cd.name as corpus, d.file, dc.cnt
  from doc as d, corp_doc as cd, doc_counts as dc
  where d.rowid = dc.doc_id and
        d.rowid = cd.doc_id
  order by cnt;

-- find sentences that start with quotes
select d.doc_id, d.file, s.sent, s.lang, d.tot_cnt
  from doc_keep as d, sent as s
  where d.doc_id = s.doc_id and
        s.lang != 'sq' and s.sent like '"%'
  limit 200;

-- get a sample of non-Albanian sentences
select d.rowid, d.file, cd.name, s.sent
  from doc as d, sent as s, corp_doc as cd
  where s.lang != 'sq' and length(s.sent) > 50 and
        d.rowid = s.doc_id and 
        d.rowid = cd.doc_id
  limit 100;

-- example of how to find bad data: first find sentences not Albanian that are
-- longer than 50 characters and only give the first 100 found
select d.rowid, d.file, cd.name, s.sent
  from doc as d, sent as s, corp_doc as cd
  where s.lang != 'sq' and length(s.sent) > 50 and
        d.rowid = s.doc_id and 
        d.rowid = cd.doc_id
  limit 100;

-- same thing, but find the highest number of non-albanian sentences; note:
-- corp_doc comes out for performance reasons (first row has doc.rowid=5841)
select d.rowid as doc_id, count(d.rowid) as cnt, d.file, s.lang, s.sent
  from doc as d, sent as s
  where s.lang != 'sq' and length(s.sent) > 50 and
        d.rowid = s.doc_id
  group by d.rowid
  order by count(d.rowid) desc
  limit 20;

-- same thing, but give the corpus name and the file with sentences having at
-- least 20 non-albanian
select d.rowid, cd.name, d.file
     from doc as d, corp_doc as cd
     where d.rowid = cd.doc_id and
           d.rowid in
        (select d.rowid
	  from doc as d, sent as s, corp_doc as cd
	  where s.lang != 'sq' and length(s.sent) > 50 and
		d.rowid = s.doc_id and 
		d.rowid = cd.doc_id
	  group by d.rowid
	  having count(d.rowid) > 20
	  limit 5);

-- doc.rowid = 5841 has 18,153 english sentences; however, there are lot (1
-- million sentences) for this document xml file, which is some wikimedia file
select d.file, count(*)
  from doc as d, sent as s
  where d.rowid = s.doc_id and d.rowid = 5841;

-- count sentence counts for this document; Albanian and English are at the top
select s.lang, count(s.lang)
  from doc as d, sent as s
  where d.rowid = s.doc_id and d.rowid = 5841
  group by s.lang
  order by count(s.lang) desc;

-- same thing but get language distributions across all document files and
-- create a table from it (this takes 26s)
drop table if exists lang_dist_by_doc;
create table lang_dist_by_doc as
    select d.rowid as doc_id, s.lang as lang, count(s.lang) as cnt
      from doc as d, sent as s
      where d.rowid = s.doc_id and s.lang is not null
      group by d.rowid, s.lang;

-- find no language and report
drop table if exists no_lang_dist_by_doc;
create table no_lang_dist_by_doc as
    select d.rowid as doc_id, count(d.rowid) as cnt
      from doc as d, sent as s
      where d.rowid = s.doc_id and s.lang is null
      group by d.rowid;

select cd.*, l.cnt
  from corp_doc as cd, no_lang_dist_by_doc as l
  where cd.doc_id = l.doc_id and cnt > 0
  order by cnt desc;


-- find the files with missing language
select d.file, l.lang, l.cnt
  from doc as d, lang_dist_by_doc as l
  where d.rowid = l.doc_id and
        l.lang is null
  order by cnt;

-- find the files that have low Albanian language counts
select d.file, l.lang, l.cnt
  from doc as d, lang_dist_by_doc as l
  where d.rowid = l.doc_id and
        l.lang = 'sq'
  order by cnt;

-- find out what's in these low Albanian files; keep in mind language detection
-- could be wrong; sentence language doesn't match lang dist language where
-- that file has both languages--this is what's happening (mixed language files)
select d.rowid, d.file, l.lang dist_lang, s.lang sent_lang, l.cnt, s.sent
  from doc as d, sent as s, lang_dist_by_doc as l
  where d.rowid = l.doc_id and
        d.rowid = s.doc_id and
        l.lang = 'sq' and l.cnt = 1
  order by d.rowid
  limit 50;

-- now we can find files with language mixes
select d.rowid, d.file, count(d.rowid) as cnt
  from doc as d, sent as s, lang_dist_by_doc as l
  where d.rowid = l.doc_id and
        d.rowid = s.doc_id and
        l.lang != s.lang and
        l.lang = 'sq'
  group by d.rowid;

-- get the proportion of Albanian to all languages per document; this isn't
-- efficient so create a table (takes 15s)
drop table if exists lang_prop_by_doc;
create table lang_prop_by_doc as
  select d.rowid as doc_id,
	 cnt as sq_cnt,
	 cnt * 1.0 / (select count(*) from sent where d.rowid = doc_id) as sq_prop,
	 (select count(*) from sent where d.rowid = doc_id and lang != l.lang) as other_cnt,
	 (select count(*) from sent where d.rowid = doc_id and lang is null) as no_lang_cnt,
	 (select count(*) from sent where d.rowid = doc_id) as tot_cnt
    from doc as d, lang_dist_by_doc as l
    where d.rowid = l.doc_id and
          l.lang = 'sq';

-- report on lang proportions and sanity check on total counts
select d.rowid, s.name, d.file, l.sq_prop, l.sq_cnt, l.tot_cnt,
  (select count(*) from sent where doc_id = d.rowid) != l.tot_cnt as bad_cnt
  from doc as d, corp_doc as s, lang_prop_by_doc as l
  where d.rowid = s.doc_id and
        l.doc_id = s.doc_id
  order by sq_prop;

-- find docs with no Albanian (in original doc set but not language proportion
-- table); these docs such as Ubuntu/xml/sq/gold.xml; these fall out of
-- lang_prop_doc because they have no occurances from the select or no language
select d.*, (select count(*) from sent where doc_id = d.rowid) as cnt
  from doc as d
  left join lang_prop_by_doc l on l.doc_id = d.rowid
  where l.rowid is null;

-- find low count docs, many Ubuntu and QED have very low counts
select d.rowid, s.name, d.file, l.sq_cnt, l.tot_cnt, l.sq_prop
  from doc as d, corp_doc as s, lang_prop_by_doc as l
  where d.rowid = s.doc_id and
        l.doc_id = s.doc_id
  order by l.tot_cnt;

-- see is the total sentence count pattern follows by corpus
select s.name, sum(l.tot_cnt) as cnt
  from doc as d, corp_doc as s, lang_prop_by_doc as l
  where d.rowid = s.doc_id and
        l.doc_id = s.doc_id
  group by s.name
  order by sum(l.tot_cnt);

-- QED is high so maybe it has low Albanian counts: not the case so we'll have
-- to go document by document
select s.name, sum(l.sq_cnt) as cnt
  from doc as d, corp_doc as s, lang_prop_by_doc as l
  where d.rowid = s.doc_id and
        l.doc_id = s.doc_id
  group by s.name
  order by sum(l.sq_cnt);

-- create sentence stats by doc
drop table if exists sent_stats_by_doc;
create table sent_stats_by_doc as
  select d.rowid as doc_id, avg(length(s.sent)) as avg, min(length(s.sent)) as min
    from doc as d, sent as s
    where d.rowid = s.doc_id
    group by d.rowid;

-- get a sample of Albanian sentences with the lowest proportion; many Albanian
-- misclassified; mostly other languages
select *
  from sent as s
  where s.doc_id in (select doc_id from lang_prop_by_doc order by sq_prop limit 2);

-- get counts for the lowest Albanian proportions: sq is in the middle
select s.lang, count(s.lang) as cnt
  from sent as s
  where s.doc_id in (select doc_id from lang_prop_by_doc order by sq_prop limit 10) and
        s.lang is not null
  group by s.lang
  order by count(s.lang);

-- get corpus names, files and stats for the lower quantile
select d.rowid, s.name, d.file, l.sq_cnt, l.sq_prop
  from doc as d, corp_doc as s, lang_prop_by_doc as l
  where d.rowid = s.doc_id and
        l.doc_id = s.doc_id and
        l.sq_prop < 0.25;

-- find the corpora with the best proportions of Albanian, so maybe remove
-- Ubuntu all completely, GNOME (249 sentences even though 70%)
select l.doc_id, cs.name, cs.url, avg(l.sq_prop) sq_prop_ave, l.sq_cnt, l.other_cnt
  from corp_doc as s, corp_src as cs, lang_prop_by_doc as l
  where l.doc_id = s.doc_id and
        s.name = cs.id
  group by s.name
  order by avg(l.sq_prop);

-- find docs based on minimum constraints; this is the query used for the final
-- corpus (see `select_doc_*` queries)
drop table if exists doc_keep;
create table doc_keep as
  select d.rowid as doc_id, cd.name, d.file, l.sq_cnt, l.tot_cnt, l.sq_prop, sd.*
    from doc as d, corp_doc as cd, lang_prop_by_doc as l, sent_stats_by_doc sd
    where d.rowid = cd.doc_id and
	  d.rowid = l.doc_id and
	  d.rowid = sd.doc_id and
	  cd.name not in ('GNOME', 'Ubuntu', 'WikiMatrix') and
	  l.sq_cnt > 50 and
	  l.sq_prop > 0.95 and
	  sd.avg > 30
    order by l.tot_cnt desc;
create index doc_keep_doc_id on doc_keep(doc_id);


---- filter and data massage at the sentence level

-- create the sentence keep table based on kept documents and classified as
-- Albanian that are not quoted (the quoted sentences seem to be from a movie
-- or book); add a count of tokens to later pair down further
drop table if exists sent_keep;
create table sent_keep as
  select s.*, length(spans) - length(replace(s.spans, '(', '')) as toklen
    from sent as s, doc_keep as d
    where d.rowid = s.doc_id and
	  lang = 'sq' and
	  s.sent not like '"%';

-- 1,474,975 sentences; update: 3,564,255; after oscar: 4,756,555
select count(*) from sent_keep;

-- pair down further by using significantly long sentences but under the model
-- token limit (not example token to word piece); the token lenght is null for
-- singletons
delete from sent_keep
  where toklen < 5 or toklen > 450 or toklen is null;

-- find gargage
select sent from sent_keep order by sent limit 10000;

-- remove gargage
delete from sent_keep
  where sent like 'e% %' escape 'e';
delete from sent_keep where sent like '/ /%';
delete from sent_keep where sent like '* CAPTCHA%';
delete from sent_keep where sent like '( Burime të%';

-- repeat until 0 row changes
update sent_keep set sent = substr(sent, 2) where sent glob '[0-9]*';

update sent_keep set sent = trim(sent, '#');
update sent_keep set sent = trim(sent, '-');
update sent_keep set sent = trim(sent, '.');
update sent_keep set sent = trim(sent, '''');
update sent_keep set sent = trim(sent, '[');
update sent_keep set sent = trim(sent, '[');
update sent_keep set sent = trim(sent, '"');
update sent_keep set sent = trim(sent, '. . .');
update sent_keep set sent = trim(sent, '$');
update sent_keep set sent = trim(sent, '*');
update sent_keep set sent = trim(sent, ',');
update sent_keep set sent = trim(sent, '-');
update sent_keep set sent = trim(sent, '+');
update sent_keep set sent = trim(sent, '&');
update sent_keep set sent = trim(sent, '%');
update sent_keep set sent = trim(sent, ')');
update sent_keep set sent = trim(sent, '/');
update sent_keep set sent = trim(sent, ':');
update sent_keep set sent = trim(sent, ';');
update sent_keep set sent = trim(sent, '< / span >');
update sent_keep set sent = trim(sent, '< / q >');
update sent_keep set sent = trim(sent, '< / h2 >');
update sent_keep set sent = trim(sent, '< / i >');
update sent_keep set sent = trim(sent, '< / cite >');
update sent_keep set sent = trim(sent);

-- remove top N gargage -- careful!
delete from sent_keep where rowid in (
  select rowid from sent_keep order by sent limit 500);

-- remove reduced sentences after cleanup
delete from sent_keep where length(sent) < 10;

-- list duplicates
select sent, toklen, count(*) as cnt
  from sent_keep
  group by sent
  order by count(*) desc;

-- delete duplicates
delete from sent_keep
    where rowid not in (
        select min(rowid)
	    from sent_keep
	    group by sent);

-- 1,289,157 sentences; update: 2,641,945; oscar: 3,984,705
select count(*) from sent_keep;

-- 26,599,075 tokens; update: 56,537,038; oscar: 121,794,474
select sum(toklen) from sent_keep;

-- oscar: 647,922,859 characters = 617.91M
select sum(length(sent)) from sent_keep;

-- split by space to get the vocabulary (this takes too long)
drop table if exists sent_keep_split;
create table sent_keep_split as
  with recursive split(rowid, value, rest) as (
     select rowid, '', sent||' ' from sent
     union all select
       rowid,
       substr(rest, 0, instr(rest, ' ')),
       substr(rest, instr(rest, ' ')+1)
       FROM split WHERE rest!='')
  select rowid, value from split where value != '';


---- clean up vocab

-- drop duplicates
drop table if exists vocab;
create table vocab as
  select distinct(value) as token
    from sent_keep_split;

-- trim left/right whitespace
update vocab set token = trim(token);

-- no number and sensical lengths
delete from vocab
  where typeof(token) != 'text';

delete from vocab
  where length(token) > 20 or length(token) < 2;

-- delete starting bad characters
delete from vocab
  where token like '%-%' or
        token like 'www%' or
        token like '%.%' or
        token like '%,%' or
        token like '% %' or
        token like '''%' or
        token like '%$%' or
        token like '%+%' or
        token like '%0%' or
        token like '%1%' or
        token like '%2%' or
        token like '%3%' or
        token like '%4%' or
        token like '%5%' or
        token like '%6%' or
        token like '%7%' or
        token like '%8%' or
        token like '%9%';

-- remove first all uppercase 3 letters
delete from vocab
  where substr(token, 1, 3) = upper(substr(token, 1, 3)) and
        length(token) > 3;

-- remove upper case only, which catches other languages as well
delete from vocab where token = upper(token);

-- only keep extended ascii only
delete from vocab where token not glob ('*[^'||char(1,255)||']*');

-- 595,816 vocabulary tokens
select count(*) from vocab;


---- cleanup
shtab

drop table sent;

create table sent as
  select doc_id, six, sid, sent, spans, toklen
         from sent_keep;

drop table sent_keep;
drop table doc_keep;
drop table lang_dist_by_doc;
drop table lang_prop_by_doc;
drop table no_lang_dist_by_doc;
drop table sent_keep_split;
drop table sent_stats_by_doc;
drop table vocab

create index sent_doc_id on sent(doc_id);
create index sent_sid on sent(sid);
create index sent_sent on sent(sent);

vacuum;


---- sanity checks

shtab


-- 2,656,763 sentences; oscar: 3,984,705
select count(*) from sent;

select * from sent limit 5000;

select * from sent order by sent limit 5000;


---- named corpus queries

-- name=select_sent_src
select d.doc_id, sr.name, sr.url, d.file, s.six, s.sid, s.sent
  from doc as d, sent as s, src as sr
  where d.doc_id = s.doc_id and d.sid = sr.rowid
  order by d.name, d.doc_id, s.six;

-- name=count_sent
select count(*) from sent;

-- name=select_sent
select sent from sent where rowid=?;

-- name=select_sent_ids
select rowid from sent;

-- name=select_sents
select sent
    from sent
    order by doc_id, six
    limit ?;

-- name=select_vocab
select * from vocab;


---- wikipedia exp

-- find docs
select rowid, * from doc where file like '%/wikipedia/%';

-- add new source
insert into corp_src (id, name, url) values ('Wikipedia', 'Wikipedia Dump', 'https://dumps.wikimedia.org');

-- corpus documents and their sources
select s.id, s.name, s.url, d.name, d.file
  from corp_src as s, corp_doc as d
  where d.sid = s.rowid and s.id = 'Wikipedia';

-- get wikipeddia source, doc, sents
select cs.id, cs.name, cs.url, cd.name, cd.file as corp_doc, d.file as doc_file_sys, s.lang, s.six, s.sid, s.sent
  from corp_src as cs, corp_doc as cd, doc as d, sent as s
  where cd.doc_id = d.rowid and
        cd.sid = cs.rowid and
        d.rowid = s.doc_id and
        cs.id = 'Wikipedia' limit 50;

-- get count of wikipedia docs 3,847,369 with 2,275,098 classified as sq
select count(*)
  from corp_src as cs, corp_doc as cd, doc as d, sent as s
  where cd.doc_id = d.rowid and
        cd.sid = cs.rowid and
        d.rowid = s.doc_id and
	s.lang = 'sq' and
        cs.id = 'Wikipedia';

-- peek at albanian sents
select s.sent, s.lang
  from corp_src as cs, corp_doc as cd, doc as d, sent as s, s.spans
  where cd.doc_id = d.rowid and
        cd.sid = cs.rowid and
        d.rowid = s.doc_id and
	s.lang = 'sq' and
        cs.id = 'Wikipedia' limit 500;

-- copy sentences to keeps
insert into sent_keep
  select s.*, 0 as toklen
    from corp_src as cs, corp_doc as cd, doc as d, sent as s
    where cd.doc_id = d.rowid and
	  cd.sid = cs.rowid and
	  d.rowid = s.doc_id and
	  s.lang = 'sq' and
	  cs.id = 'Wikipedia';

-- update token length
update sent_keep set toklen = length(spans) - length(replace(spans, '(', ''));

select sent, lang, 0 as toklen
  from sent limit 50;

-- find wikipedia docs
select s.*
  from doc as d, sent as s
  where d.file like '%/wikipedia/%' and
        d.rowid = s.doc_id limit 500;


---- oscar

-- add new source
insert into corp_src (id, name, url) values (
       'oscar', 'OSCAR', 'https://oscar-project.github.io/documentation/versions/oscar-2301');

-- corpus docs
select cs.name, cs.id, cd.file, d.dir_id as directory_id, d.file, cd.doc_id
  from corp_src as cs, corp_doc as cd, doc as d
  where cd.name = cs.id and
        cd.doc_id = d.rowid and
        cs.id = 'oscar';

-- get a sample
select cs.name, cs.id, cd.file, d.dir_id as directory_id, d.file, cd.doc_id, s.sent
  from corp_src as cs, corp_doc as cd, doc as d, sent as s
  where cd.name = cs.id and
        cd.doc_id = d.rowid and
	cd.doc_id = s.doc_id and
        cs.id = 'oscar'
	limit 100;

-- get count 2,099,792; after cleanup: 1,340,766
select count(*)
  from corp_src as cs, corp_doc as cd, doc as d, sent as s
  where cd.name = cs.id and
        cd.doc_id = d.rowid and
	cd.doc_id = s.doc_id and
        cs.id = 'oscar';


---- reporting

-- stats

-- name=select_sent_count
select count(*) from sent;

-- name=select_token_count
select sum(toklen) from sent;

-- name=select_char_count
select sum(length(sent)) from sent;

-- name=select_corpus_sources
select cs.id as name, cs.url, count(*) as count
  from corp_src as cs, corp_doc as cd, doc as d, sent as s
  where cd.name = cs.id and
        cd.doc_id = d.rowid and
	cd.doc_id = s.doc_id
  group by cs.id;
