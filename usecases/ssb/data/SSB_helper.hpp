#pragma once

#include <string>
#include <vector>
#include <map>


constexpr uint32_t CUST_MORTALITY = 3;
constexpr uint32_t SPARSE_BITS = 2;
constexpr uint32_t SPARSE_KEEP = 3;

static const std::vector<std::string> COLOR = {"almond", "antique", "aquamarine", "azure", "beige", "bisque", "black", "blanched", "blue",
    "blush", "brown", "burlywood", "burnished", "chartreuse", "chiffon", "chocolate", "coral",
    "cornflower", "cornsilk", "cream", "cyan", "dark", "deep", "dim", "dodger", "drab", "firebrick",
    "floral", "forest", "frosted", "gainsboro", "ghost", "goldenrod", "green", "grey", "honeydew",
    "hot", "indian", "ivory", "khaki", "lace", "lavender", "lawn", "lemon", "light", "lime", "linen",
    "magenta", "maroon", "medium", "metallic", "midnight", "mint", "misty", "moccasin", "navajo",
    "navy", "olive", "orange", "orchid", "pale", "papaya", "peach", "peru", "pink", "plum", "powder",
    "puff", "purple", "red", "rose", "rosy", "royal", "saddle", "salmon", "sandy", "seashell", "sienna",
    "sky", "slate", "smoke", "snow", "spring", "steel", "tan", "thistle", "tomato", "turquoise", "violet",
    "wheat", "white", "yellow"};

static const std::vector<std::string> TYPE_SYL1 = {"STANDARD","SMALL","MEDIUM","LARGE","ECONOMY","PROMO"};
static const std::vector<std::string> TYPE_SYL2 = {"ANODIZED","BURNISHED","PLATED","POLISHED","BRUSHED"};
static const std::vector<std::string> TYPE_SYL3 = {"TIN","NICKEL","BRASS","STEEL","COPPER"};
static const std::vector<std::vector<std::string>> TYPE = {TYPE_SYL1,TYPE_SYL2,TYPE_SYL3};


static const std::vector<std::string> CONT_SYL1 = {"SM","LG","MEDIUM","JUMBO","WRAP"};
static const std::vector<std::string> CONT_SYL2 = {"CASE","BOX","BAG","JAR","PKG","PACK","CAN","DRUM"};
static const std::vector<std::vector<std::string>> CONT = {CONT_SYL1,CONT_SYL2};

static const std::vector<std::string> SEGMENT = {"AUTOMOBILE","BUILDING","FURNITURE","MACHINERY","HOUSEHOLD"};

static const std::vector<std::string> PRIORITY = {"1-URGENT","2-HIGH","3-MEDIUM","4-NOT SPECIFIED","5-LOW"};
static const std::vector<std::string> SHIPMODE = {"REG AIR","AIR","RAIL","SHIP","TRUCK","MAIL","FOB"};

static const std::vector<std::string> REGION = {"AFRICA","AMERICA","ASIA","EUROPE","MIDDLE EAST"};
static const std::map<std::string, int32_t> NATION = {
                                {"ALGERIA",0},
                                {"ARGENTINA",1},
                                {"BRAZIL",1},
                                {"CANADA",1},
                                {"EGYPT",4},
                                {"ETHIOPIA", 0},
                                {"FRANCE",3},
                                {"GERMANY",3},
                                {"INDIA", 2},
                                {"INDONESIA",2},
                                {"IRAN",4},
                                {"IRAQ",4},
                                {"JAPAN", 2},
                                {"JORDAN",4},
                                {"KENYA", 0},
                                {"MOROCCO",0},
                                {"MOZAMBIQUE",0},
                                {"PERU",1},
                                {"CHINA",2},
                                {"ROMANIA",3},
                                {"SAUDI ARABIA",4},
                                {"VIETNAM", 2},
                                {"RUSSIA",3},
                                {"UNITED KINGDOM",3},
                                {"UNITED STATES", 1}
                                                };

static uint32_t partkey_to_price(uint64_t p){
    uint32_t price = 90000;
    price += (p/10) % 20001;        /* limit contribution to $200 */
    price += (p % 1000) * 100;
    
    return price;
}


