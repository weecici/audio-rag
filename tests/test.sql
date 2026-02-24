SELECT    *
FROM      cs431;

SELECT    *
FROM      cs431_df;

SELECT    *
FROM      cs431_pl;

DROP      TABLE if EXISTS cs431_pl;

DROP      TABLE if EXISTS cs431_df;

DROP      TABLE if EXISTS cs431;

SELECT    file_name,
          text
FROM      cs431
WHERE     doc_len > 100;